# Results and uncertainty

Clinical result context and software verification are separate artifacts.
No model weights are included. No domain-adaptation comparator training code is included.

## Included result context

- `ct_dataset_reported.csv` / `.tex`: reported Table I CT cohort inventory.
- `mri_dataset_statistics.json`, `mri_dataset_tables.tex` and the three table images:
  verified aggregate native-geometry statistics of the recovered MRI subsets.
- `ours_5shot_run_means.csv`: deidentified scalar run means for direct Swin UNETR LoRA
  and Bridge-Tuning LoRA, with center/fold/seed keys and source-result hashes.
- `ours_5shot_historical_summary.json` / `.tex`: verified legacy means and sample SD over
  available fold×seed run means. Center counts are retained, including 4/5-fold groups.
- `table2_reported_context.csv`: all 160 own-model adaptation cells from the audited
  historical Table II, with source-match status and coverage counts. It is reported
  context; seven means remain unresolved in that audit and are flagged explicitly.
- `table2_source_verified.tex`: the portion of Table II whose mean and SD can be
  recovered from matching raw run records; unverified cells are displayed as unavailable.
- `demo/`: synthetic software metrics and predictions, not clinical table numbers.

Legacy Table II also contains pooled three-seed groups without outer CV; these cannot
be relabeled as 3×5-fold experiments. The audited legacy CV cells retained 4 or 5 folds
in different groups and include missing runs. Some legacy evaluations excluded low-Dice
or failed cases. The release evaluates every case and reports errors explicitly, so
rerunning it is not guaranteed to recreate means from that filtered evaluator.

## Metric definition

Foreground Dice uses `2*intersection/(prediction+reference)` with smoothing 1e-5.
HD95 uses scipy binary erosion (default connectivity) to obtain mask surfaces, computes
both directed Euclidean distances with `distance_transform_edt` and takes the 95th
percentile of their concatenation. Metrics are computed after deterministic spacing
and foreground cropping, before inverse transform to the native image grid.

Default HD95 units are **voxels in the resampled grid**. The grid is anisotropic, so
this is not a physical millimeter distance. `--hd95-unit mm` uses actual voxel spacing
in the Euclidean distance calculation. Unit changes must be stated when comparing data.

Both masks empty: Dice=1 and HD95=0. Exactly one empty: near-zero Dice and HD95=100 in
the selected distance unit, with empty flags in per-case output. The finite penalty is
a declared convention for undefined surface distance, not a measured distance.
There is no low-Dice exclusion and no silent fallback to HD95=0 when scipy is absent.

## Aggregation unit

`evaluate.py` writes the mean and sample SD **across cases**, with ddof=1. This is not
the variation across seeds. When fewer than two cases are present, SD is `null`, not 0.

`aggregate_results.py` expects one mean per held-out run, keyed by center/method/fold/seed.
It averages equal-weight folds within each seed and then reports mean/sample SD across
seeds. Fold counts must be consistent within each group. An Average row cannot be
formed by simply averaging center SDs; form each seed's center-average first and take
its SD if complete matched center/seed data are available.

```bash
python aggregate_results.py --csv results/ours_5shot_run_means.csv \
  --output runs/seed_summary.json
```

The historical LoRA tables used SD across available fold×seed runs. That statistic is
included under its correct label separately from the seed-level aggregate generated
above. Historical direct and bridge HNCH groups do not have identical fold keys, so
they cannot be passed directly to a paired test without a registered common subset.

## Optional paired uncertainty

For **newly registered matching runs**, use:

```bash
python aggregate_results.py --csv runs/matched_run_means.csv --reference direct \
  --output runs/paired_uncertainty.json
```

The tool checks complete matching fold/seed keys, uses seed-level fold-averaged paired
differences, returns a 95% t interval and an exact two-sided sign-flip p-value. Matching
keys alone do not prove that the underlying held-out subjects match; verify registrations.
The t interval assumes approximately normal independent seed-level differences. With
only three pairs, the smallest two-sided sign-flip p-value is 0.25. This tool therefore
does not claim conventional statistical significance merely from three favorable seeds.
Overlapping CV folds are not treated as independent replicates.

## Bridge-selection equations

For features `Z_C` from target C and `Z_B` from bridge B:

```
D(C -> B) = mean over z in Z_C of min over g in Z_B ||z-g||_2
W(Z) = sum_i ||z_i - mean(Z)||_2^2 / (N-1)
Delta W(B,C) = W(Z_B) - W(Z_C)
```

Closest-D minimizes the directional support distance; Coverage-DeltaW maximizes the
coverage gap. The release implements these definitions directly. It does not copy the
archived Gaussian mean/std surrogate as if it were the same criterion. Historical
selection scalar values are not regenerated by this code-preparation check.
