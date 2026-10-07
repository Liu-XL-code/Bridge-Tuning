# Executed release checks

The executable checks use synthetic data in an isolated directory. No clinical paper
experiments were retrained and no source scans, logs or experiment settings were edited.
The machine-readable receipt is `results/demo/release_verification.json`.

## Passed checks

1. **Nine core tests**: LoRA merge equivalence; encoder/decoder trainability for the four
   strategies; directional support distance and sample covariance coverage; fixed
   subject folds and seeded support sampling; seed-level aggregation and paired-key
   checks; CT geometry/intensity/label transforms; MRI label rules and constant LR;
   label-free manifest reading; Dice/HD95 units, empty masks, overlap and stopping rules.
2. **All thirteen YAML profiles** passed configuration validation.
3. **Reduced-width end-to-end run**: feature_size=12, ROI=64³, two bridge epochs and two
   target epochs, explicit validation/test partitions, final target selection,
   sliding-window inference and inverse native-grid restoration.
4. **Full architecture training check**: feature_size=48, ROI=96³, foundation-initialized
   FFT bridge training followed by LoRA target adaptation, two epochs per stage,
   gradient accumulation with one resident patch, validation and held-out testing.
5. **Retained checkpoint inference**: a separately retained FFT bridge was used on an LPS-oriented
   phantom, restoring native shape/affine and binary output. Separately, a retained
   PUCH LoRA variant (not included in this source-only edition) passed the same inference/evaluation
   checks. Its additional check is labeled separately in the receipt.
6. **Frozen feature extraction and selector calculation**: one fixed foundation
   checkpoint, image-only features, D/DeltaW and no label-file reads.
7. **Eleven command-line help entry points**: train, evaluate, infer, split generation,
   CV driver, statistical aggregation, features, selectors, M&Ms frame preparation,
   geometry audit and visualization.
8. **Checkpoint exports**: initialization loads all 157 foundation non-head tensors;
   a separately retained FFT bridge loads its 159 tensors strictly. Additional ablation states
   were checked separately and are not included in this edition.
   The final file inventory contains SHA-256 hashes.

The phantom test is not a measure of clinical accuracy. The included montage and
prediction use a separately retained FFT bridge on synthetic data. Additional nonbundled
checkpoint checks in the receipt are software checks, not paper result estimates.

## Verified environment

Python 3.10.0; PyTorch 2.11.0+cu130; MONAI 1.4.0; NumPy 1.26.4; scipy 1.15.3;
nibabel 5.4.2; PyYAML 6.0.3; matplotlib 3.10.9; einops 0.8.2. Checks used CUDA.
Dependency deprecation warnings from MONAI 1.4 / recent PyTorch were emitted, with
successful forwards, backward passes and inference. `requirements-verified.txt`
records the package versions; select the matching CUDA wheel for the installed GPU.

## Practical limits

These checks establish that the release code trains, validates, tests and infers with
the intended model architecture and geometry. They do not rerun 200/300-epoch clinical
training, regenerate MRI selection tables or claim exact reproduction of legacy
metrics with filtered cases. Historical result coverage and checkpoint identities are
recorded separately. Private/public source images and annotations are not included.

## Source-only packaging

No model checkpoint binaries are included. The checks above were executed before
packaging; retained-model checks used separately available models. The executable
training, evaluation and inference source is unchanged from those checks. Documentation
now starts with a weight-free synthetic run that creates its own checkpoints.
