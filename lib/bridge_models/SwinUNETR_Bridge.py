import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class SwinUNETRLinear_Prob(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=2,
        feature_size=48,
        use_checkpoint=True,
        freeze_encoder=True,
        freeze_decoder=False,
        **kwargs
    ):
        super().__init__()
        self.model = SwinUNETR(
                in_channels=in_channels,
                out_channels=out_channels,
                feature_size=feature_size,
                use_checkpoint=use_checkpoint
        )
        if freeze_encoder:
            self._freeze_encoder()
        if freeze_decoder:
            self._freeze_decoder()

    def _freeze_encoder(self):
        # 如需“线性探测”（只训线性头），通常应该冻结编码器（swinViT）和 encoder* 路径
        for name, p in self.model.named_parameters():
            top = name.split(".")[0]
            if top == "swinViT" or ("encoder" in top):
                p.requires_grad = False

    def _freeze_decoder(self):
        to_freeze_top_names = {"out"}
        for name, p in self.model.named_parameters():
            top = name.split(".")[0]
            if ("decoder" in top) or (top in to_freeze_top_names):
                p.requires_grad = False


    def forward(self,x):
        return self.model(x)