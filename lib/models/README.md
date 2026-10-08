# Segmentation Models

This directory wraps MONAI Swin UNETR for bridge-domain pre-adaptation and few-shot target adaptation. The model classes are exported from `lib.models` and selected by `model_name` in the training configuration.

| File | Exported class | Adaptation strategy |
| --- | --- | --- |
| `swin_unetr.py` | `SwinUNETRModel` | Full-parameter fine-tuning. |
| `swin_unetr_lora.py` | `SwinUNETRLoRA` | Low-rank updates to encoder attention layers, with a fully trainable decoder. |
| `swin_unetr_bitfit.py` | `SwinUNETRBitFit` | Encoder bias and normalization parameters remain trainable; the decoder is fully trainable by default. |
| `swin_unetr_linear_prob.py` | `SwinUNETRLinear_Prob` | Freezes the encoder while leaving the decoder trainable by default. |

## LoRA

`SwinUNETRLoRA` supports the encoder attention `qkv` and `proj` linear layers selected through `target_modules`. Its default rank is 8. The original attention weights remain frozen, and the low-rank matrices are trained during adaptation.

Keep `decoder_lora=False`. Decoder LoRA is not implemented and `decoder_lora=True` is unsupported. The supported setting fully fine-tunes the decoder and output head.

Pretrained and adapted checkpoints are supplied by the user. No model weights are included in this repository. See the root README and `configs/` for the two-stage training commands.
