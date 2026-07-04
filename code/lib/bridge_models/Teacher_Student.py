import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR


ESO_IDX = 5
TAU = 0.25    # ROI 阈值
BETA, GAMMA = 1.0, 0.2 # 蒸馏损失的权重
"""
class CollapseHead(nn.Module):

    def __init__(self, in_channels=30, out_channels=2):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
"""
class CollapseHead(nn.Module):
    """坍缩头：将30通道映射到2通道（深度残差版本）"""
    def __init__(self, in_channels=30, out_channels=2, hidden_channels=64):
        super().__init__()
        # 主分支：三层非线性
        self.main_branch = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # 输出层
        self.output = nn.Conv3d(hidden_channels, out_channels, kernel_size=1)

        # 快捷连接
        self.shortcut = nn.Conv3d(in_channels, hidden_channels, kernel_size=1)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.main_branch(x)
        out = out + identity
        out = self.output(out)
        return out

# 参数量 ≈ 8,000-10,000（更强的非线性能力）

def roi_weight(p_eso):
    """ROI加权函数，基于食管概率"""
    # p_eso: [B,1,D,H,W] 概率
    return BETA * torch.sigmoid((p_eso - TAU) / 0.08) + GAMMA


class SwinUNETR_Teacher(nn.Module):
    """教师网络：冻结的30通道SwinUNETR"""
    def __init__(
        self,
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=30,
        feature_size=48,
        use_checkpoint=True,
        **kwargs
    ):
        super().__init__()
        self.model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, x):
        return self.model(x)

    def load_pretrained(self, checkpoint_path):
        """加载预训练权重"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)

        # 处理state_dict命名
        new_state_dict = {}
        for k, v in state_dict.items():
            key = k[7:] if k.startswith('module.') else k
            if key.startswith('model.'):
                key = key[6:]  # 移除'model.'前缀
            new_state_dict[key] = v

        self.model.load_state_dict(new_state_dict, strict=False)
        print(f"Teacher loaded from {checkpoint_path}")

        # 冻结所有参数
        for param in self.parameters():
            param.requires_grad = False
        self.eval()


class SwinUNETR_Student(nn.Module):
    """学生网络：2通道SwinUNETR（全参数训练）"""
    def __init__(
        self,
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=2,
        feature_size=48,
        use_checkpoint=True,
        **kwargs
    ):
        super().__init__()
        self.model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, x):
        return self.model(x)

    def load_backbone_from_teacher(self, teacher_state_dict):
        """从教师网络加载主干权重（忽略输出头）"""
        student_dict = self.model.state_dict()

        # 只加载主干和编码器/解码器权重，跳过输出头
        pretrained_dict = {
            k: v for k, v in teacher_state_dict.items()
            if k in student_dict and 'out' not in k
        }

        student_dict.update(pretrained_dict)
        self.model.load_state_dict(student_dict, strict=False)
        print(f"Student backbone loaded from teacher ({len(pretrained_dict)} layers)")


class TeacherStudentBridge(nn.Module):
    """教师-学生桥接网络

    支持两种训练模式：
    - 阶段1：训练坍缩头（stage='collapse'）
    - 阶段2：知识蒸馏训练学生（stage='distill'）
    """
    def __init__(
        self,
        img_size=(96, 96, 96),
        in_channels=1,
        teacher_out_channels=30,
        student_out_channels=2,
        feature_size=48,
        use_checkpoint=True,
        teacher_pretrained_path=None,
        stage='collapse',  # 'collapse' 或 'distill'
        **kwargs
    ):
        super().__init__()
        self.stage = stage
        self.eso_idx = ESO_IDX

        # 教师网络（冻结）
        self.teacher = SwinUNETR_Teacher(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=teacher_out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint
        )

        if teacher_pretrained_path:
            self.teacher.load_pretrained(teacher_pretrained_path)

        # 坍缩头
        self.collapse_head = CollapseHead(
            in_channels=teacher_out_channels,
            out_channels=student_out_channels
        )

        # 学生网络（仅在蒸馏阶段需要）
        if stage == 'distill':
            self.student = SwinUNETR_Student(
                img_size=img_size,
                in_channels=in_channels,
                out_channels=student_out_channels,
                feature_size=feature_size,
                use_checkpoint=use_checkpoint
            )
            # 从教师加载主干权重
            self.student.load_backbone_from_teacher(self.teacher.model.state_dict())
        else:
            self.student = None

    def set_stage(self, stage):
        """切换训练阶段"""
        assert stage in ['collapse', 'distill'], "stage must be 'collapse' or 'distill'"
        self.stage = stage

        if stage == 'distill' and self.student is None:
            raise ValueError("Student network not initialized. Recreate model with stage='distill'")

        # 设置参数训练状态
        if stage == 'collapse':
            # 阶段1：只训练坍缩头
            for param in self.collapse_head.parameters():
                param.requires_grad = True
            if self.student is not None:
                for param in self.student.parameters():
                    param.requires_grad = False
        else:
            # 阶段2：冻结坍缩头，训练学生
            for param in self.collapse_head.parameters():
                param.requires_grad = False
            for param in self.student.parameters():
                param.requires_grad = True

    def forward(self, x):
        """前向传播

        返回：
        - 阶段1 (collapse): (collapse_logits, p_eso, teacher_logits)
        - 阶段2 (distill): (student_logits, collapse_logits, p_eso, teacher_logits)
        """
        # 教师前向传播（冻结）
        with torch.no_grad():
            teacher_logits = self.teacher(x)  # [B, 30, D, H, W]
            teacher_probs = F.softmax(teacher_logits, dim=1)
            p_eso = teacher_probs[:, self.eso_idx:self.eso_idx+1]  # [B, 1, D, H, W]

        if self.stage == 'collapse':
            # 阶段1：训练坍缩头
            collapse_logits = self.collapse_head(teacher_logits)  # [B, 2, D, H, W]
            return collapse_logits, p_eso, teacher_logits

        else:  # stage == 'distill'
            # 阶段2：知识蒸馏
            with torch.no_grad():
                collapse_logits = self.collapse_head(teacher_logits)  # [B, 2, D, H, W]

            student_logits = self.student(x)  # [B, 2, D, H, W]
            return student_logits, collapse_logits, p_eso, teacher_logits

    def get_trainable_params(self):
        """获取可训练参数"""
        if self.stage == 'collapse':
            return self.collapse_head.parameters()
        else:
            return self.student.parameters()