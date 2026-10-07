# Local dataset preparation

The CT experiment uses TCGA-ESCA as bridge cohort B and the five private target cohorts
PUCH, HMUCC, FAZZU, HNCH and HNCH-Late as C. HNCH-Late is treated independently.
No images, labels, case identifiers, split files, feature arrays, predictions or
weights are included. Obtain access to private cohorts through the data owners.
Public TCGA-ESCA acquisition information: https://www.cancerimagingarchive.net/collection/tcga-esca/

Create local CSV manifests with columns `id,subject_id,image,label`. These are schema
names only; the repository includes no CSV files. Each row identifies one authorized
3D NIfTI image/mask pair. Resolve relative paths against the CSV location, and use
pseudonymous subject identifiers. Repeated scans from one subject share subject_id.
Images and masks must have matching shapes and physical grids, with millimeter units.
`audit_geometry.py --csv YOUR_LOCAL_MANIFEST` checks them without altering headers.

For B, provide disjoint train and optional validation CSVs. For C, `make_splits.py`
fixes five subject-level test folds with split_seed=1024. Seeds 0, 1 and 2 independently
sample K support subjects from the other four folds and control training randomness.
Unused training-pool subjects are not added to that fold's test partition. No target
validation set is created. The generator defines the release's prospective split
protocol; a seed alone does not reconstruct unavailable historical split registrations.
