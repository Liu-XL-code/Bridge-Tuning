#!/usr/bin/env python
"""Bridge-domain ranking from frozen encoder features.

Each input .npz file must contain an array named "features" with shape
(n_samples, n_features). The script computes:

- Sample Support Distance: average nearest-neighbor distance from target
  features to bridge features. Smaller is better.
- Distribution Coverage Gap: trace(cov_bridge) - trace(cov_target).
  Larger is better.
"""

import argparse
import csv
from pathlib import Path

import numpy as np


def load_features(path):
    arr = np.load(path)
    if "features" not in arr:
        raise KeyError(f"{path} does not contain key 'features'")
    features = np.asarray(arr["features"], dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"{path}: expected shape (n_samples, n_features), got {features.shape}")
    return features


def coverage(features):
    if features.shape[0] < 2:
        return 0.0
    centered = features - features.mean(axis=0, keepdims=True)
    return float((centered * centered).sum() / (features.shape[0] - 1))


def sample_support_distance(target, bridge, chunk_size=512):
    distances = []
    bridge_norm = (bridge * bridge).sum(axis=1, keepdims=True).T
    for start in range(0, target.shape[0], chunk_size):
        query = target[start:start + chunk_size]
        query_norm = (query * query).sum(axis=1, keepdims=True)
        dist2 = np.maximum(query_norm + bridge_norm - 2.0 * query @ bridge.T, 0.0)
        distances.append(np.sqrt(dist2.min(axis=1)))
    return float(np.concatenate(distances).mean())


def normalize(values, higher_is_better):
    values = np.asarray(values, dtype=np.float64)
    if np.allclose(values.max(), values.min()):
        return np.ones_like(values)
    scaled = (values - values.min()) / (values.max() - values.min())
    return scaled if higher_is_better else 1.0 - scaled


def parse_bridge_arg(item):
    if "=" not in item:
        path = Path(item)
        return path.stem, str(path)
    name, path = item.split("=", 1)
    return name, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-features", required=True)
    parser.add_argument("--bridge-features", nargs="+", required=True)
    parser.add_argument("--out", default="results/bridge_selection.csv")
    args = parser.parse_args()

    target = load_features(args.target_features)
    target_coverage = coverage(target)

    rows = []
    for item in args.bridge_features:
        name, path = parse_bridge_arg(item)
        bridge = load_features(path)
        support = sample_support_distance(target, bridge)
        gap = coverage(bridge) - target_coverage
        rows.append({
            "bridge": name,
            "path": path,
            "sample_support_distance": support,
            "distribution_coverage_gap": gap,
        })

    support_rank = normalize([r["sample_support_distance"] for r in rows], higher_is_better=False)
    gap_rank = normalize([r["distribution_coverage_gap"] for r in rows], higher_is_better=True)
    for row, s_score, g_score in zip(rows, support_rank, gap_rank):
        row["rank_score"] = float(0.5 * s_score + 0.5 * g_score)

    rows.sort(key=lambda r: r["rank_score"], reverse=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bridge",
                "path",
                "sample_support_distance",
                "distribution_coverage_gap",
                "rank_score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    for idx, row in enumerate(rows, start=1):
        print(
            f"{idx:02d}. {row['bridge']}: "
            f"D={row['sample_support_distance']:.4f}, "
            f"gap={row['distribution_coverage_gap']:.4f}, "
            f"score={row['rank_score']:.4f}"
        )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
