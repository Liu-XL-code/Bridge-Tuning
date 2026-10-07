# Training and split protocol

## Main CT pipeline

| Setting | A -> B, `ct_bridge.yaml` | B -> C, `ct_target_lora.yaml` |
|---|---|---|
| Backbone | Swin UNETR, feature_size=48 | Same |
| Input/output channels | 1 / 2 | 1 / 2 |
| Strategy | Full-parameter fine-tuning | Encoder LoRA + full decoder/head |
| LoRA | None | r=8, alpha=16, dropout=0.1, `qkv` attention projections |
| Encoder convolution blocks | Trainable | Frozen |
| Loss | MONAI DiceCE, one-hot target, softmax | Same |
| Optimizer | AdamW | AdamW |
| Initial LR | 1e-4 | 5e-4 |
| Weight decay | 0.01 | 1e-5 |
| Bias/one-dimensional decay | Excluded | Excluded |
| Maximum epochs | 200 | 300 |
| LR schedule | 10-epoch linear warmup + cosine | 20-epoch linear warmup + cosine |
| Volume batch | 2 | 4 |
| Patches/volume | 2 | 2 |
| Maximum effective patch batch | 4 | 8 |
| Resident microbatch | 2 patches | 2 patches |
| Precision default | FP32 | FP32 |
| Loss-based early stopping | Disabled | Enabled, rules below |
| Checkpoint selection | Final by default; optional validation Dice | Final; no target validation in few-shot CV |

Microbatch gradients are weighted by microbatch size and accumulated before one AdamW
step. A smaller final effective batch is kept (`drop_last=False`). Random initialization
is allowed for debugging, but the proposed method starts from the foundation backbone.
`--init` checks all non-head tensors; an unexpected architecture mismatch is an error.
It can merge a native LoRA release checkpoint before inserting fresh target adapters.

A historical FFT bridge (not included here) used its recorded best-validation selection, which
is different from the release default final-epoch bridge recipe. Set `selection: val_dice`
and supply a disjoint bridge validation CSV to use that explicit rule.

## Strategy definitions

- `fft`: update all parameters.
- `linear_prob`: historical project name for a frozen encoder with the **entire decoder
  and segmentation head trainable**; it is not a classifier-only linear probe.
- `bitfit`: train encoder bias and normalization parameters, plus the full decoder/head.
- `lora`: train LoRA A/B attention matrices while freezing encoder base parameters;
  the full decoder/head remains trainable.

MRI LoRA preserves the separate historical MRI implementation: `qkv` and `proj`,
dropout=0, and encoder convolution blocks outside Swin remain trainable.
These differences are configuration fields rather than hidden modality-specific code.

## MRI optimization

MRI bridge/target profiles use AdamW LR=1e-4, weight decay=1e-5 on **all** trainable
parameters, constant LR, no warmup, maximum 100 epochs, volume batch=1 and two patches.
Mixed precision is enabled on CUDA, and loss-plateau patience is 10. Final selection
matches the independent archived public-MRI training driver.

## Termination and checkpoint selection

CT target training tracks per-epoch mean loss. After at least 40 epochs, it smooths
losses with a 10-epoch moving average. It stops when the decrease from
`smoothed[-30]` to the latest smoothed loss is less than 1% of `smoothed[-30]`.
This exact indexing mirrors the archived implementation; the comparison spans 29
smoothed-index increments. MRI uses `smoothed[-10]` after at least 20 epochs.
Increasing loss also satisfies the stopping condition. No validation/test metric
participates in this stopping rule.

`selection: final` selects the last completed epoch, including an early-stopped epoch.
`selection: val_dice` instead selects the highest mean held-out validation Dice
(strict `>` improvement), measured every 10 epochs plus the final epoch. It requires
a validation CSV and is not used for target CV without a validation set.
`final.pt`, optional `best.pt` and `selected.pt` make the choice explicit.

The training function receives training and optional validation items only. It never
receives test items. `training_receipt.json`, `split_ids.json`, `history.json` and the
resolved `config.json` expose the realized protocol for every run.

## Target folds and seeds

`make_splits.py` shuffles distinct subjects with `split_seed=1024` and creates five
balanced held-out folds. Each fold's test subjects stay fixed across support seeds
0, 1 and 2. For each support seed, K distinct subjects are sampled from the other four
folds; one volume per selected subject is used for adaptation. The same support seed
also controls training/augmentation randomness. There is no separate target validation
set in this few-shot protocol. All runs must use identical registered folds when paired.

The legacy experiments retained four or five folds in different groups. Their observed
counts are included with historical results. A new five-fold split cannot repair a
missing historical fold or prove it was used in an old experiment.

## Unlabeled target-image access

Bridge selection may use all available target images, including held-out test images
without labels, to characterize the target-center feature distribution. This is a
transductive selection setting. Frozen feature extraction and `select_bridge.py` do
not read label files. Supervised target adaptation is restricted to K labeled volumes.
For an inductive protocol, restrict the feature manifest to permitted non-test images
and state that change explicitly when reporting results.

## Practical example

For a five-shot single-fold target run initialized from your trained FFT bridge:

```bash
python make_splits.py --csv data/target.csv --shots 5 --folds 5 --seeds 0 1 2 \
  --split-seed 1024 --output runs/target_splits
python train.py --config configs/ct_target_lora.yaml \
  --train-csv runs/target_splits/fold_0_seed_0_train.csv \
  --init runs/ct_bridge/selected.pt --seed 0 --role target \
  --output runs/target_fold0_seed0 --device auto
python evaluate.py --checkpoint runs/target_fold0_seed0/selected.pt \
  --csv runs/target_splits/fold_0_test.csv --partition test \
  --output runs/target_fold0_seed0/test --device auto
```
