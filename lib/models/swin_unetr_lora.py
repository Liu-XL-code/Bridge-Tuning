"""
SwinUNETR with LoRA (Low-Rank Adaptation)
支持对编码器和解码器分别控制是否使用LoRA参数高效微调

使用方式：
1. 通过配置文件（推荐）：
   在 configs/finetuning_only.yaml 中设置：
   model_name: SwinUNETRLoRA
   peft_config:
     r: 64
     alpha: 16
     dropout: 0.1
     target_modules: ['qkv', 'proj']
     freeze_encoder: False  # False=使用LoRA, True=完全微调
     freeze_decoder: False  # False=使用LoRA, True=完全微调

   训练器会自动传递 peft_config，无需修改训练器代码

2. 直接初始化：
   model = SwinUNETRLoRA(
       in_channels=1,
       out_channels=2,
       feature_size=48,
       encoder_lora=True,   # 编码器使用LoRA
       decoder_lora=False,  # 解码器完全微调
       lora_r=64,
       lora_alpha=16
   )

参数说明：
- freeze_encoder=False: 编码器使用LoRA（参数高效）
- freeze_encoder=True:  编码器完全微调（所有参数可训练）
- freeze_decoder同理
"""

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class LoRALayer(nn.Module):
    """
    LoRA层：在原始Linear层基础上添加低秩分解
    forward(x) = W(x) + (W_b @ W_a)(x) * scaling
    """
    def __init__(self, original_layer, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha

        # 计算缩放因子
        self.scaling = alpha / rank

        # 🔧 获取原始层的设备，确保新创建的层在同一设备上
        device = next(original_layer.parameters()).device

        # 低秩矩阵 A 和 B
        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)

        # 🔧 将新创建的层移动到与原始层相同的设备
        self.lora_a = self.lora_a.to(device)
        self.lora_b = self.lora_b.to(device)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 初始化：A使用高斯初始化，B使用零初始化
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

        # 冻结原始层
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

    def forward(self, x):
        # 原始输出 + LoRA输出
        result = self.original_layer(x)
        lora_result = self.lora_b(self.lora_a(self.dropout(x))) * self.scaling
        return result + lora_result


class SwinUNETRLoRA(nn.Module):
    """
    SwinUNETR with LoRA

    参数说明：
    - encoder_lora: True时编码器使用LoRA，False时编码器完全微调
    - decoder_lora: True时解码器使用LoRA，False时解码器完全微调

    支持两种初始化方式：
    1. 直接传参: SwinUNETRLoRA(encoder_lora=True, lora_r=64, ...)
    2. 通过peft_config: SwinUNETRLoRA(peft_config={'r': 64, 'freeze_encoder': False, ...})
    """
    def __init__(
        self,
        in_channels=1,
        out_channels=104,
        feature_size=48,
        use_checkpoint=True,
        encoder_lora=True,
        decoder_lora=False,
        lora_r=None,
        lora_alpha=None,
        lora_dropout=None,
        target_modules=None,
        peft_config=None,
        **kwargs
    ):
        super().__init__()

        # 如果提供了 peft_config，从中提取参数
        if peft_config is not None and isinstance(peft_config, dict):
            # freeze_encoder=False 表示使用LoRA，True表示完全微调
            if encoder_lora is None:
                encoder_lora = not peft_config.get('freeze_encoder', False)
            if decoder_lora is None:
                decoder_lora = not peft_config.get('freeze_decoder', True)
            if lora_r is None:
                lora_r = peft_config.get('r', 8)
            if lora_alpha is None:
                lora_alpha = peft_config.get('alpha', 16)
            if lora_dropout is None:
                lora_dropout = peft_config.get('dropout', 0.1)
            if target_modules is None:
                target_modules = peft_config.get('target_modules', ['qkv'])

        # 设置默认值
        if encoder_lora is None:
            encoder_lora = True
        if decoder_lora is None:
            decoder_lora = False
        if lora_r is None:
            lora_r = 8
        if lora_alpha is None:
            lora_alpha = 16
        if lora_dropout is None:
            lora_dropout = 0.1
        if target_modules is None:
            target_modules = ['qkv']

        # 保存模型配置（用于重置模型）
        self._model_config = {
            'img_size': (96, 96, 96),
            'in_channels': in_channels,
            'out_channels': out_channels,
            'feature_size': feature_size,
            'use_checkpoint': use_checkpoint
        }

        # 创建基础模型
        self.model = SwinUNETR(**self._model_config)

        # 保存LoRA配置（先不应用，等加载权重后再应用）
        self.encoder_lora = encoder_lora
        self.decoder_lora = decoder_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules

        # 标记：LoRA是否已应用
        self._lora_applied = False

        print(f"\n=== SwinUNETR LoRA 模型创建 ===")
        print(f"编码器策略: {'LoRA (r={}, alpha={})'.format(lora_r, lora_alpha) if encoder_lora else '完全微调'}")
        print(f"解码器策略: {'LoRA (r={}, alpha={})'.format(lora_r, lora_alpha) if decoder_lora else '完全微调'}")
        print(f"注意: LoRA 将在加载预训练权重后自动应用\n")

    def _apply_strategy(self):
        """
        应用微调策略（设置参数的 requires_grad）

        注意：LoRA 层的转换在 load_state_dict 中已完成
        这里只需要设置哪些参数可训练
        """
        # 1. 先冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 2. 处理编码器
        if self.encoder_lora:
            # LoRA模式：只有 lora_a 和 lora_b 可训练
            trainable_count = 0
            for name, param in self.model.named_parameters():
                if 'swinViT' in name or 'encoder' in name:
                    if 'lora_a' in name or 'lora_b' in name:
                        param.requires_grad = True
                        trainable_count += 1
            print(f"  ✓ 编码器: LoRA模式，{trainable_count} 个LoRA参数可训练")
        else:
            # 完全微调模式
            self._unfreeze_encoder()

        # 3. 处理解码器
        if self.decoder_lora:
            # 解码器LoRA（通常不需要）
            self._apply_lora_to_decoder()
        else:
            # 解码器完全微调
            self._unfreeze_decoder()

    def _apply_lora_to_encoder(self):
        """
        对编码器应用LoRA

        重要：此方法会将已加载权重的 qkv 层替换为 LoRALayer
        原始的 qkv 层（包含预训练权重）会被保留在 LoRALayer.original_layer 中
        """
        if not self.encoder_lora:
            return 0

        lora_count = 0

        # 遍历 swinViT 的所有层
        for layer_idx in range(1, 5):  # layers1, layers2, layers3, layers4
            layer_name = f'layers{layer_idx}'
            if not hasattr(self.model.swinViT, layer_name):
                continue

            layer = getattr(self.model.swinViT, layer_name)
            basic_layer = layer[0]  # BasicLayer

            # 遍历所有blocks
            for block_idx, block in enumerate(basic_layer.blocks):
                # 对 qkv 应用 LoRA
                if hasattr(block.attn, 'qkv') and 'qkv' in self.target_modules:
                    original_qkv = block.attn.qkv  # 这个层已经加载了预训练权重！

                    # 创建 LoRA 层（会保留 original_qkv 的权重）
                    lora_qkv = LoRALayer(
                        original_qkv,
                        rank=self.lora_r,
                        alpha=self.lora_alpha,
                        dropout=self.lora_dropout
                    )

                    # 替换层
                    setattr(block.attn, 'qkv', lora_qkv)
                    lora_count += 1

                # 对 proj 应用 LoRA
                if hasattr(block.attn, 'proj') and 'proj' in self.target_modules:
                    original_proj = block.attn.proj  # 这个层已经加载了预训练权重！

                    lora_proj = LoRALayer(
                        original_proj,
                        rank=self.lora_r,
                        alpha=self.lora_alpha,
                        dropout=self.lora_dropout
                    )

                    setattr(block.attn, 'proj', lora_proj)
                    lora_count += 1

        if lora_count > 0:
            print(f"  ✓ 已将 {lora_count} 个 attention 层转换为 LoRA 结构")
            print(f"  ✓ 预训练权重已保留在 original_layer 中")

        return lora_count

    def _apply_lora_to_decoder(self):
        """对解码器应用LoRA（可选功能）"""
        # 解码器通常使用卷积层，LoRA主要用于线性层
        # 这里提供一个基础实现，可以根据需要扩展
        lora_count = 0

        # 遍历解码器的所有模块
        for name, module in self.model.named_modules():
            # 只处理解码器部分
            if not name.startswith('decoder'):
                continue

            # 如果是 Linear 层且名称匹配 target_modules
            if isinstance(module, nn.Linear):
                # 可以在这里添加 LoRA
                # 但通常解码器不需要 LoRA
                pass

        if lora_count > 0:
            print(f"  解码器: 已应用 {lora_count} 个 LoRA 层")
        else:
            print(f"  解码器: 未应用 LoRA (解码器主要是卷积层)")

    def _unfreeze_encoder(self):
        """解冻编码器所有参数（完全微调）"""
        param_count = 0
        for name, param in self.model.named_parameters():
            top = name.split('.')[0]
            if top == 'swinViT' or 'encoder' in top:
                param.requires_grad = True
                param_count += 1
        print(f"  编码器: 已解冻 {param_count} 个参数")

    def _unfreeze_decoder(self):
        """解冻解码器所有参数（完全微调）"""
        param_count = 0
        for name, param in self.model.named_parameters():
            top = name.split('.')[0]
            if 'decoder' in top or top == 'out':
                param.requires_grad = True
                param_count += 1
        print(f"  解码器: 已解冻 {param_count} 个参数")

    def _print_trainable_params(self):
        """打印可训练参数统计"""
        trainable_params = 0
        all_params = 0

        for name, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        percentage = 100 * trainable_params / all_params if all_params > 0 else 0

        print(f"\n可训练参数统计:")
        print(f"  可训练: {trainable_params:,} ({percentage:.2f}%)")
        print(f"  总参数: {all_params:,}")
        print(f"  冻结参数: {all_params - trainable_params:,}\n")

    def forward(self, x):
        """前向传播"""
        return self.model(x)

    def load_state_dict(self, state_dict, strict=True):
        """
        重写 load_state_dict 实现先加载后替换策略

        策略：
        1. 检测是否为预训练权重格式（无 .original_layer）
        2. 如果是，先重置为标准SwinUNETR，再加载，再应用LoRA
        3. 如果不是（已经是LoRA格式），直接加载
        """
        # 检测是否为预训练权重格式
        is_pretrained_format = False
        for key in state_dict.keys():
            if '.attn.qkv.weight' in key and '.original_layer' not in key:
                is_pretrained_format = True
                break

        if is_pretrained_format:
            print("\n=== 检测到预训练权重格式，采用先加载后替换策略 ===")

            # 如果已经应用过LoRA，先重置为标准SwinUNETR
            if self._lora_applied:
                print("检测到模型已有LoRA结构，正在重置为标准SwinUNETR...")

                # 🔧 保存当前设备信息，确保新模型在相同设备上
                try:
                    device = next(self.model.parameters()).device
                except StopIteration:
                    # 如果模型没有参数，使用默认设备
                    device = torch.device('cpu')

                self.model = SwinUNETR(**self._model_config)

                # 🔧 将新模型移动到相同设备
                self.model = self.model.to(device)

                self._lora_applied = False
                print(f"  ✓ 模型已重置为标准结构（设备: {device}）")

            print("步骤 1/3: 加载预训练权重到标准SwinUNETR结构...")

            # 过滤掉输出层（类别数可能不同）
            filtered_state_dict = {}
            skipped_count = 0
            for key, value in state_dict.items():
                if 'out.conv' in key or 'classifier' in key:
                    skipped_count += 1
                    continue
                filtered_state_dict[key] = value

            if skipped_count > 0:
                print(f"  - 跳过 {skipped_count} 个输出层参数（类别数不同）")

            incompatible = super().load_state_dict(filtered_state_dict, strict=False)

            # 过滤日志中的输出层警告
            missing = [k for k in incompatible.missing_keys if 'out.conv' not in k and 'classifier' not in k]
            unexpected = [k for k in incompatible.unexpected_keys if 'classifier' not in k]

            if missing:
                print(f"  ⚠ 缺失键: {len(missing)} 个 (输出层类别数不同，属正常)")
            if unexpected:
                print(f"  ⚠ 多余键: {len(unexpected)} 个 (预训练模型的classifier)")

            print("  ✓ 预训练权重加载完成")

            # 步骤2: 将qkv层替换为LoRALayer（权重会自动从已加载的层复制）
            print("\n步骤 2/3: 将attention层替换为LoRA结构...")
            self._apply_lora_to_encoder()

            # 步骤3: 应用微调策略
            print("\n步骤 3/3: 应用微调策略...")
            self._apply_strategy()

            # 标记LoRA已应用
            self._lora_applied = True

            # 打印最终统计
            self._print_trainable_params()
            print("=== LoRA模型初始化完成 ===\n")

            return incompatible
        else:
            # 不是预训练权重，或LoRA已应用，使用标准加载
            return super().load_state_dict(state_dict, strict=strict)



def create_swin_unetr_lora(
    in_channels=1,
    out_channels=104,
    feature_size=48,
    use_checkpoint=True,
    peft_config=None,
    **kwargs
):
    """
    创建 SwinUNETR LoRA 模型的辅助函数

    Args:
        peft_config: 字典，包含 LoRA 配置
            - freeze_encoder: False表示编码器使用LoRA，True表示完全微调
            - freeze_decoder: False表示解码器使用LoRA，True表示完全微调
            - r: LoRA rank
            - alpha: LoRA alpha
            - dropout: LoRA dropout
            - target_modules: 目标模块列表
    """
    if peft_config is None:
        peft_config = {}

    # 从 peft_config 提取参数
    # 注意：freeze_encoder=False 表示使用LoRA，True表示完全微调
    encoder_lora = not peft_config.get('freeze_encoder', False)
    decoder_lora = not peft_config.get('freeze_decoder', True)

    lora_r = peft_config.get('r', 8)
    lora_alpha = peft_config.get('alpha', 16)
    lora_dropout = peft_config.get('dropout', 0.1)
    target_modules = peft_config.get('target_modules', ['qkv'])

    return SwinUNETRLoRA(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        use_checkpoint=use_checkpoint,
        encoder_lora=encoder_lora,
        decoder_lora=decoder_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        **kwargs
    )
