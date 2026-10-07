# Code tree

```text
BridgeTuning/
├── README.md                     # English quick start and all entry points
├── LICENSE / NOTICE              # Source license and upstream provenance
├── requirements*.txt
├── pyproject.toml
├── train.py                      # Bridge training or target adaptation
├── evaluate.py                   # Fixed-checkpoint validation/test
├── infer.py                      # Native-grid binary NIfTI prediction
├── make_splits.py                # Fixed subject folds and K-shot support seeds
├── run_cv.py                     # Complete fold × seed target runs
├── aggregate_results.py          # Seed-level mean/SD and optional paired uncertainty
├── extract_features.py           # Frozen, image-only Swin features
├── select_bridge.py              # Directional D and covariance-trace DeltaW
├── prepare_mnms.py               # Corresponding annotated 3D cine frame
├── audit_geometry.py             # Read-only image/label grid audit
├── create_example.py             # Synthetic NIfTI generation
├── visualize.py                  # Native-grid image/label/prediction montage
├── run_smoke.py                  # Small complete software demonstration
├── convert_checkpoint.py         # Explicit locally trusted legacy conversion
├── src/bridge_tuning/
│   ├── config.py                 # Validated, resolved configuration
│   ├── model.py                  # Swin UNETR, strategies, checkpoints/LoRA merging
│   ├── data.py                   # Manifests, geometry, CT/MRI transforms and labels
│   ├── engine.py                 # AdamW, accumulation, stopping, evaluation/inversion
│   ├── metrics.py                # Dice, symmetric HD95 and case-level aggregation
│   ├── splits.py                 # Grouped subject partition registration
│   └── selection.py              # Label-free feature criteria
├── configs/                      # CT, M&Ms, CC-359, SAML and smoke profiles
├── checkpoints/                   # Weight-free placeholder; instructions for external checkpoints
├── examples/synthetic/           # Twelve image/mask pairs and disjoint CSV manifests
├── assets/                       # Workflow SVG and example montage
├── results/                      # Aggregate historical context + synthetic receipts
├── docs/                         # Detailed preprocessing, data, model and run rules
└── tests/                        # Core invariants, split/selector/statistics checks
```

## How the entry points connect

`train.py` calls manifest/geometry validation, shared deterministic transforms and
train-only augmentation, then builds Swin UNETR and applies the selected strategy.
The trainer writes the resolved config, split IDs, history, selected checkpoint and
training receipt. `evaluate.py` and `infer.py` load the same checkpoint-embedded image
profile and run sliding-window prediction. Evaluation computes metrics on the
preprocessed grid; saved masks undo spatial transforms to the original NIfTI grid.

`run_cv.py` registers subject-level test folds once and then calls training with only
K support items for each fold/seed. It tests after selection and records one scalar
run mean per metric. `aggregate_results.py` distinguishes case, fold and seed variation.

The feature branch is independent of segmentation labels. `extract_features.py` uses
the same frozen checkpoint for every domain; `select_bridge.py` reads the feature
arrays and selects the smallest D or largest DeltaW. No external comparator pipeline
or external experiment trainer is required to import or run this package.
