# Preprocessing and labels

Training, validation, test and inference call the same deterministic preprocessing
implementation in `src/bridge_tuning/data.py`. Inference reads images only and
uses the profile stored inside the checkpoint.

## Deterministic segmentation transforms

| Operation | CT esophageal tumor | MRI |
|---|---|---|
| Input | Static 3D, one-channel NIfTI; mm coordinates | Static 3D, one-channel NIfTI; mm coordinates |
| Orientation | RAS | RAS |
| Voxel spacing | 1.5 × 1.5 × 2.0 mm | 1.0 × 1.0 × 1.5 mm |
| Image interpolation | MONAI `bilinear` in 3D (trilinear sampling) | Same |
| Mask interpolation | Nearest neighbor | Nearest neighbor |
| Intensity transform | Clip HU to [-175,250], then `(HU+175)/425` | Clip/scale using each volume's 0.5th and 99.5th percentiles |
| Scaled intensity | [0,1] before augmentation | [0,1] |
| Foreground crop | Bounding box of positive normalized **image** voxels | Same |
| Train-only padding | Pad all ROI axes to at least 96 | Same |
| Random train crop | 96³, 2 samples/volume, pos:neg center weights 1:1 | Same |
| Evaluation | Entire cropped volume, sliding windows | Same |

The foreground crop uses the image, not the tumor annotation. CT positive normalized
voxels correspond to HU greater than -175; this is not a label-driven tumor bounding
box. Interpolation and rounding can affect the exact resampled size. No whole-volume
96³ resize is applied in segmentation inference. Padding used by sliding-window
inference is handled automatically and removed by the inferer.

Positive/negative sampling defines probabilities; it does not guarantee exactly one
positive and one negative crop for every pair. Labels may contain no foreground; MONAI
then samples eligible background centers rather than manufacturing tumor pixels.

## Binary label definitions

| Profile | Foreground rule | Output |
|---|---|---|
| CT | Original label > 0 | Esophageal tumor |
| M&Ms | Original integer label == 1 | LV blood pool only |
| CC-359 | STAPLE hippocampus probability >= 0.5 | Hippocampus union |
| SAML | Original label > 0 | Whole prostate, union of annotated zones |

The `label_mode`, `label_values` and `label_threshold` settings implement these rules
after nearest-neighbor spacing. CC-359 can accept a probability mask directly. Do not
use a generic `>0` rule for a STAPLE probability mask. For an already binarized CC-359
mask, the 0.5 threshold gives the same result.

M&Ms 4D inputs must first be reduced to a corresponding annotated 3D image/label frame:

```bash
python prepare_mnms.py --image data/cine4d.nii.gz --label data/mask4d.nii.gz \
  --max-lv-frame --output-image prepared/image3d.nii.gz \
  --output-label prepared/label3d.nii.gz
```

The project preparation selected the annotated frame with maximal label-1 voxel count
as an ED proxy. The helper records this rule and does not claim anatomical ED verification.
Use `--frame INDEX` when the actual annotated frame is known. Repeated frames from one
subject must share `subject_id` during splitting.

## Train-only augmentation

Each spatial axis is independently flipped with probability 0.3. A random 90-degree
rotation is applied with probability 0.3 (`max_k=3`). CT additionally applies intensity
scaling of ±0.1 and shifting of ±0.1, independently with probability 0.2.
Intensity augmentation can move values outside [0,1]; the deterministic input remains
normalized. MRI does not use those CT intensity augmentations.

## Geometry checks and native-space output

Image and label must have identical 3D shapes and matching NIfTI affine matrices
within 0.001 mm elementwise tolerance. Declared spatial units must be mm or unspecified
(unspecified units are treated as mm, consistent with the archived preparation).
The pipeline rejects a mismatch before training. It does not overwrite NIfTI headers.

The recovered SAML RUNMC subset has image/label origin differences of about 1.77–2.01 mm
in z. The release checker reports these until alignment has been reviewed and corrected
in a derived copy. Equal pixel-array size alone cannot establish physical alignment.

The network predicts in the RAS/resampled/cropped grid. MONAI inverse transforms undo
the crop, spacing and orientation with nearest-neighbor mask interpolation. The saved
binary `uint8` prediction has the input image's original shape, affine and spatial header.
Dice/HD95 are evaluated in the preprocessed grid before that inverse transform;
native-space masks are visualization/inference outputs, not a different metric space.

## Feature branch

`extract_features.py` reads no label files. It applies checkpoint image preprocessing,
or an explicit `--config` image profile (use the MRI profile for MRI),
then area-resizes the complete cropped volume to the ROI and globally pools the frozen
Swin stage-3 output. This is distinct from random patch training and sliding-window
segmentation. The release uses a consistent RAS-before-spacing order; historical
feature scripts used separate preprocessing branches, so old published criterion
values are not represented as regenerated by this release.
