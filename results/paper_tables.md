# Paper Results

This file records the main reported numbers so reviewers can compare their reproduced runs against the paper. Dice is reported as mean +/- std over three seeds. HD95 is in voxel units.

## 5-shot LoRA: Direct PEFT vs Bridge-Tuning

| Method | Center-1 Dice | Center-1 HD95 | Center-2 Dice | Center-2 HD95 | Center-3 Dice | Center-3 HD95 | Center-4 Dice | Center-4 HD95 | Center-5 Dice | Center-5 HD95 | Avg Dice | Avg HD95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A -> C direct PEFT | 0.540 +/- 0.10 | 31.7 | 0.514 +/- 0.07 | 32.9 | 0.588 +/- 0.06 | 45.3 | 0.588 +/- 0.06 | 49.2 | 0.520 +/- 0.05 | 45.6 | 0.550 | 40.9 |
| Bridge-Tuning A -> B -> C | 0.585 +/- 0.10 | 23.5 | 0.582 +/- 0.05 | 26.0 | 0.644 +/- 0.08 | 42.7 | 0.668 +/- 0.03 | 20.6 | 0.554 +/- 0.06 | 40.9 | 0.607 | 30.7 |

## 5-shot From-scratch Baselines

| Method | Center-1 Dice | Center-2 Dice | Center-3 Dice | Center-4 Dice | Center-5 Dice | Avg Dice |
|---|---:|---:|---:|---:|---:|---:|
| UNETR from scratch | 0.213 +/- 0.06 | 0.093 +/- 0.09 | 0.044 +/- 0.08 | 0.019 +/- 0.02 | 0.030 +/- 0.03 | 0.080 |
| nnU-Net from scratch | 0.585 +/- 0.01 | 0.359 +/- 0.04 | 0.610 +/- 0.07 | 0.626 +/- 0.08 | 0.451 +/- 0.03 | 0.526 |
| A -> C direct PEFT | 0.540 +/- 0.10 | 0.514 +/- 0.07 | 0.588 +/- 0.06 | 0.588 +/- 0.06 | 0.520 +/- 0.05 | 0.550 |
| Bridge-Tuning A -> B -> C | 0.585 +/- 0.10 | 0.582 +/- 0.05 | 0.644 +/- 0.08 | 0.668 +/- 0.03 | 0.554 +/- 0.06 | 0.607 |

## 5-shot Domain-adaptation Baselines

| Method | Center-1 Dice | Center-2 Dice | Center-3 Dice | Center-4 Dice | Center-5 Dice | Avg Dice |
|---|---:|---:|---:|---:|---:|---:|
| CORAL | 0.518 +/- 0.03 | 0.428 +/- 0.03 | 0.477 +/- 0.04 | 0.531 +/- 0.08 | 0.326 +/- 0.07 | 0.456 |
| MMD | 0.476 +/- 0.05 | 0.367 +/- 0.02 | 0.380 +/- 0.10 | 0.509 +/- 0.03 | 0.324 +/- 0.05 | 0.411 |
| DANN | 0.517 +/- 0.03 | 0.412 +/- 0.08 | 0.514 +/- 0.02 | 0.575 +/- 0.05 | 0.359 +/- 0.03 | 0.475 |
| MIC | 0.521 +/- 0.04 | 0.398 +/- 0.04 | 0.487 +/- 0.04 | 0.566 +/- 0.05 | 0.339 +/- 0.02 | 0.462 |
| PMTrans | 0.501 +/- 0.03 | 0.341 +/- 0.07 | 0.483 +/- 0.01 | 0.579 +/- 0.04 | 0.320 +/- 0.02 | 0.445 |
| Bridge-Tuning A -> B -> C | 0.585 +/- 0.10 | 0.582 +/- 0.05 | 0.644 +/- 0.08 | 0.668 +/- 0.03 | 0.554 +/- 0.06 | 0.607 |
