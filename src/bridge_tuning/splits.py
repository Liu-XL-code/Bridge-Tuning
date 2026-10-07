"""Fixed subject-level folds, independently seeded K-volume support sampling."""
import csv
import random
from pathlib import Path

from .config import write_json
from .data import assert_disjoint, read_manifest


def write_manifest(path, cases):
    path = Path(path)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "subject_id", "image", "label"])
        writer.writeheader()
        for case in cases:
            writer.writerow({k: case[k] for k in writer.fieldnames})


def make_folds(manifest, folds, k, seeds, split_seed, output):
    cases = read_manifest(manifest)
    grouped = {}
    for case in cases:
        grouped.setdefault(case["subject_id"], []).append(case)
    subjects = sorted(grouped)
    if not 2 <= folds <= len(subjects) or k < 1:
        raise ValueError("Need >=2 folds, at least one subject per fold, and K>=1")
    random.Random(split_seed).shuffle(subjects)
    sizes = [len(subjects) // folds + (i < len(subjects) % folds) for i in range(folds)]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    start, report = 0, []
    for fold, size in enumerate(sizes):
        test_subjects = set(subjects[start:start + size])
        start += size
        test = [c for c in cases if c["subject_id"] in test_subjects]
        pool = [s for s in subjects if s not in test_subjects]
        if len(pool) < k:
            raise ValueError("K exceeds the number of distinct eligible training subjects")
        write_manifest(output / f"fold_{fold}_test.csv", test)
        for seed in seeds:
            generator = random.Random(seed)
            selected = generator.sample(sorted(pool), k)
            # One support volume per sampled subject, even when ED/ES or repeat scans are indexed.
            training = [generator.choice(grouped[s]) for s in selected]
            assert_disjoint(training, test)
            write_manifest(output / f"fold_{fold}_seed_{seed}_train.csv", training)
            report.append({"fold": fold, "support_seed": seed, "train_ids": [c["id"] for c in training],
                           "test_ids": [c["id"] for c in test], "train_subjects": selected,
                           "test_subjects": sorted(test_subjects)})
    write_json(output / "split_registration.json", {"fold_split_seed": split_seed,
        "folds": folds, "k": k, "support_seeds": seeds, "target_validation": "none",
        "test_partition_fixed_across_support_seeds": True, "splits": report})
    return report
