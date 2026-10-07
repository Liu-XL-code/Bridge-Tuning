# CT preprocessing

Training, validation, testing and inference share `src/bridge_tuning/data.py`.
The three CT YAML profiles make the relevant parameters explicit.

1. Load a static, one-channel 3D NIfTI image and corresponding mask; require matching
   shapes and affine matrices within 0.001 mm elementwise tolerance.
2. Reorient both to RAS.
3. Resample to 1.5 × 1.5 × 2.0 mm: trilinear image sampling (MONAI bilinear in 3D),
   nearest-neighbor mask sampling.
4. Clip image intensities to [-175,250] HU and map `(HU+175)/425` to [0,1].
5. Binarize tumor labels using original label > 0.
6. Crop image and mask to the bounding box of positive normalized image voxels.
   This uses image foreground, not the tumor annotation.
7. For training only, pad as needed and sample two 96³ patches per volume with
   positive:negative center weights 1:1. This does not guarantee one of each per pair.

Train-only augmentation flips each spatial axis with probability 0.3, applies a
random 90-degree rotation with probability 0.3, and independently scales and shifts
intensities by ±0.1 with probability 0.2. Augmented values may exceed [0,1].

Evaluation uses the entire cropped volume and sliding-window inference, overlap=0.5.
Segmentation does not resize the whole volume to 96³. Inference reverses crop,
spacing and orientation with nearest-neighbor mask sampling, restoring the input
shape, affine and spatial header. The output is a binary uint8 NIfTI mask.

Dice and HD95 are calculated in the resampled/cropped grid. HD95 defaults to voxel
units; `evaluate.py --hd95-unit mm` explicitly requests physical distances.
If both masks are empty, Dice=1 and HD95=0. If only one is empty, the configured
finite HD95 penalty is 100 in the requested units. Zero-Dice cases are retained.

The feature branch reads no label files, resizes the complete normalized foreground
to the configured ROI, and globally pools frozen Swin stage-3 features. This is
separate from segmentation patch sampling and sliding-window inference.
