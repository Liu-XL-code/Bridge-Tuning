# A -> B -> C training

| Parameter | A -> B | B -> C |
|---|---|---|
| Backbone | Swin UNETR, feature_size=48, 1 input / 2 output channels | Same |
| Update | Full-parameter fine-tuning | Swin qkv LoRA, full decoder/head |
| LoRA | None | rank=8, alpha=16, dropout=0.1 |
| Encoder convolutions | Trainable | Frozen |
| Loss | MONAI DiceCE, one-hot labels, softmax | Same |
| Optimizer | AdamW | AdamW |
| Initial learning rate | 1e-4 | 5e-4 |
| Weight decay | 0.01 | 1e-5 |
| Warmup | 10 epochs | 20 epochs |
| Schedule | Linear warmup then cosine | Same |
| Maximum epochs | 200 | 300 |
| Volume batch | 2 | 4 |
| Patches per volume | 2 | 2 |
| Resident patch microbatch | 2 | 2 |
| Precision | FP32 | FP32 |
| Early stopping | Disabled | Training-loss plateau |
| Checkpoint selection | Final by default; optional val_dice | Final |

Bias and one-dimensional normalization parameters are excluded from weight decay.
Microbatch gradients are weighted by their sizes and accumulated for one optimizer
step per volume batch; the last partial batch is kept. Use `a_foundation.yaml` for
conversion of a separately obtained trusted initializer, `b_bridge.yaml` to train B,
and `c_target.yaml` to adapt B on the K labeled target volumes. A pretraining is not
performed here. Initialization validates all non-head tensors; it cannot silently
accept an incompatible backbone. LoRA checkpoints are loaded with their adapters.

Target training tracks mean loss per epoch. After at least 40 epochs, it compares
the latest 10-epoch smoothed loss with smoothed[-30], stopping when the relative
decrease is below 1%. This exact indexing spans 29 smoothed-index increments;
increasing loss also meets the condition. No validation/test metric participates.
`selection: final` uses the last completed epoch, including an early-stopped epoch.

An explicit `selection: val_dice` chooses the highest mean disjoint validation Dice
at the configured interval plus final epoch, with strict improvement. It requires
a validation CSV and is not used by the target no-validation few-shot protocol.
`final.pt`, optional `best.pt`, and `selected.pt` record the choice locally. Generated
history, resolved configuration, split IDs and training receipt remain local.

Use identical fixed test folds across the three support seeds for paired runs. Do
not treat all folds × seeds as independent subjects. `evaluate.py` reports per-case
values and case-level mean/sample SD for its evaluated partition; that SD is not a
seed-level SD. Aggregate completed fold means within each seed before reporting
uncertainty across seeds. A single value has no estimable sample SD.

Frozen-feature assessment may use all target images without their labels in the
transductive setting. Supervised target adaptation only sees the K selected labels.
