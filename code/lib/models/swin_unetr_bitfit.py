import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class SwinUNETRBitFit(nn.Module):
    """
    当预训练模型当中不包含下游任务数据集时采用Encoder BitFit策略，但是解码器完全开放微调
    当与训练模型已经包含了下游任务的数据集时采用Encoder Deocoder BitFit策略，解码器完全开放微调
    """
    def __init__(
        self,
        in_channels=1,
        out_channels=104,
        feature_size=48,
        use_checkpoint=True,
        encoder_bitfit=True,
        decoder_bitfit=False,
        **kwargs
    ):
        super().__init__()
        self.model = SwinUNETR(
                img_size=(96, 96, 96),
                in_channels=in_channels,
                out_channels=out_channels,
                feature_size=feature_size,
                use_checkpoint=use_checkpoint
        )


        self.encoder_bitfit = encoder_bitfit
        self.decoder_bitfit = decoder_bitfit
        self._apply_bitfit()

    def _apply_bitfit(self):
        """
        BitFit策略：
        - 编码器：只保留bias和norm参数可训练（BitFit）
        - 解码器：完全可训练
        """
        for name, param in self.model.named_parameters():
            top = name.split(".")[0]

            # 判断是否属于编码器
            is_encoder = (top == "swinViT") or ("encoder" in top)
            is_decoder = (top == "decoder") or ("decoder" in top)
            if is_encoder and self.encoder_bitfit:
                # 编码器：应用 BitFit
                # LayerNorm 和 BatchNorm 的参数全部保持可训练
                if any(norm_type in name for norm_type in ['norm', 'bn', 'layernorm', 'batchnorm']):
                    param.requires_grad = True
                # bias 参数保持可训练
                elif 'bias' in name:
                    param.requires_grad = True
                # 其他 weight 参数冻结
                else:
                    param.requires_grad = False
            elif is_decoder and self.decoder_bitfit:
                # 解码器和输出层：完全可训练
                if any(norm_type in name for norm_type in ['norm', 'bn', 'layernorm', 'batchnorm']):
                    param.requires_grad = True
                # bias 参数保持可训练
                elif 'bias' in name:
                    param.requires_grad = True
                # 其他 weight 参数冻结
                else:
                    param.requires_grad = False
    def forward(self,x):
        return self.model(x)