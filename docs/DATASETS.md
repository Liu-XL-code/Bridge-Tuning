# Dataset sources and data preparation

The release contains synthetic NIfTI examples only. Acquire public datasets from their
providers and prepare anonymous manifests locally. Private cohort scans/annotations
are not redistributed.

## CT esophageal tumor task

The original study uses a project cohort labeled TCGA-ESCA as the public bridge and
PUCH, HMUCC, FAZZU, HNCH and HNCH-Late as institutional targets. HNCH-Late is a separate
acquisition domain after scanner replacement. Labels represent esophageal tumors.

- Public collection entry: [TCIA TCGA-ESCA](https://www.cancerimagingarchive.net/collection/tcga-esca/).
- The archived paper dataset table describes larger cohort inventories; the recovered
  executable subset contains 42 image/mask pairs per CT cohort.
- A separately retained historical FFT bridge used 30 train / 6 validation / 6 held-out volumes
  within its 42-volume TCGA-labeled subset.
- These are **project volume counts**, not independently verified unique-patient counts
  in the public TCIA collection. Downloading that collection does not automatically
  reconstruct the project's selected series, derived volumes or tumor annotations.
- The private annotations and the exact TCGA project mapping are not contained in this
  anonymous ZIP. Populate `subject_id` from a valid subject-to-series mapping before
  subject-level CV; do not equate a file count with a patient count.

Table I context is included in `results/ct_dataset_reported.csv` with its original
reported volume/slice values. It is labeled as reported context, not a new inventory
scan or proof that the entire original inventory is available in this release.

## M&Ms cardiac MRI

- Provider: [M&Ms challenge](https://www.ub.edu/mnms/).
- Primary description: [M&Ms challenge paper](https://diposit.ub.edu/items/2883c0a6-1a05-4607-a098-a4ffea750136).
- The recovered experimental index includes **345** subjects: vendor/domain A=95,
  B=125, C=75 and D=50 (Siemens, Philips, GE and Canon).
- One corresponding annotated 3D cine frame per subject is prepared by selecting
  maximum LV label-1 volume as an ED proxy. This differs from counting every cine frame
  or the complete provider inventory.
- The target is LV blood pool (`label==1`), not all cardiac structures together.
- `prepare_mnms.py` extracts an image/label frame; `mnms_*.yaml` applies the correct label
  rule and MRI preprocessing. Use a shared subject ID if more than one frame is included.

## Calgary-Campinas-359

- Provider and access: [CC-359 download page](https://www.ccdataset.com/download).
- Original multi-vendor, multi-field-strength T1 brain MRI; domains are vendor × field
  strength: GE 1.5T / 3T, Philips 1.5T / 3T, Siemens 1.5T / 3T.
- The project task is **hippocampus** segmentation using the STAPLE probability masks
  thresholded at `>=0.5`, rather than the separate whole-brain skull-stripping benchmark.
- The recovered index includes **353** compatible pairs, with domain counts
  60 / 58 / 56 / 60 / 60 / 59. In the recovered copy, three images lack a hippocampus
  mask and three have incompatible image/mask dimensions. These local exclusions are
  distinct from the provider's stated two failed hippocampus masks.
- Use `cc359_*.yaml`; it accepts the probability mask and thresholds it explicitly.
- The provider states CC BY-ND 4.0 for the dataset. This ZIP provides only links and
  aggregate statistics, not modified copies of those scans or masks.

## SAML multi-site prostate MRI

- Provider and prepared-data link: [SAML dataset page](https://liuquande.github.io/SAML/).
- Source families are NCI-ISBI 2013, I2CVB and PROMISE12, linked by the provider.
- The recovered project index contains **116** volumes across BIDMC=12, BMC=30,
  HK=12, I2CVB=19, RUNMC=30 and UCL=13.
- The target is whole prostate (`label>0`), including the union of zonal labels where
  present. Use `mri_*.yaml`.
- The distributed provider copy is already prepared to 384×384 in the axial plane;
  this is distinct from the release's subsequent 1×1×1.5 mm resampling and ROI sampling.
- RUNMC origin differences in the recovered copy must be reviewed before paired
  image/label processing. `audit_geometry.py` reports them without changing source files.

## Verified aggregate MRI statistics

The three tables in `results/mri_dataset_tables.tex`, the accompanying images and
`results/mri_dataset_statistics.json` use NIfTI geometry from the recovered project
indexes. Slice count is native axis-2 depth; the spacing row is native axis-2 voxel
spacing, **not DICOM slice thickness**. Mean anatomical volume uses the stated binary
foreground rule multiplied by native voxel volume. It is computed before the release's
resampling or image crop. See table notes for frame selection and excluded pairs.

## Splits and access conditions

Use distinct bridge/target domains for transfer. Register subject-level target folds
before K-shot sampling; keep test subjects fixed across support seeds. The trainer
uses only the supplied training manifest and optional separate validation manifest.
Bridge selection can use unlabeled held-out target images in a clearly stated
transductive protocol. Never include test labels in feature extraction or checkpoint
selection. Follow each provider's data-access and attribution conditions.
