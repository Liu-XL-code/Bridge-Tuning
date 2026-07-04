"""
Swin UNETR Model for Medical Image Segmentation
使用MONAI的Swin UNETR实现
"""

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class SwinUNETRModel(nn.Module):
    """
    Swin UNETR模型包装器
    """

    def __init__(
        self,
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=104,
        feature_size=48,
        use_checkpoint=True,
        **kwargs
    ):
        """
        Args:
            img_size: 输入图像尺寸 (H, W, D) - 注意：MONAI的SwinUNETR不需要此参数
            in_channels: 输入通道数
            out_channels: 输出类别数
            feature_size: 特征维度
            use_checkpoint: 是否使用梯度检查点（节省显存）
        """
        super().__init__()

        self.model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, x):
        """前向传播"""
        return self.model(x)

    def load_from_checkpoint(self, checkpoint_path):
        """从检查点加载模型"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # 处理DataParallel/DistributedDataParallel的state_dict
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # 移除'module.'前缀
            else:
                new_state_dict[k] = v

        self.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded model from {checkpoint_path}")
