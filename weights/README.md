# Model Weights

Large model files are intentionally not tracked by git.

Expected local layout:

```text
weights/
|-- foundation_swinunetr.pth
|-- bridge_tcga_esca_swinunetr.pth
`-- target_lora_checkpoint.pth
```

Use the MONAI Swin UNETR implementation together with the foundation checkpoint source specified by the paper. Stage-1 bridge checkpoints and Stage-2 target checkpoints can be placed here for inference.
