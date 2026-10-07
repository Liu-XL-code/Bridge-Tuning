"""Rank candidate bridges using frozen, unlabeled image features (.npy)."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import numpy as np
from bridge_tuning.config import write_json
from bridge_tuning.selection import assess

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", required=True, help="Target features, N by D")
    p.add_argument("--bridge", required=True, action="append", help="NAME=feature_file.npy; repeat for candidates")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    target = np.load(a.target, allow_pickle=False)
    results = {}
    for spec in a.bridge:
        name, separator, file = spec.partition("=")
        if not separator or not name or name in results:
            raise ValueError("Use distinct bridge names in NAME=file.npy form")
        results[name] = assess(target, np.load(file, allow_pickle=False))
    report = {"criteria": results, "closest_D": min(results, key=lambda n: results[n]["D_target_to_bridge"]),
              "coverage_delta_W": max(results, key=lambda n: results[n]["delta_W"]),
              "labels_accessed": False, "target_image_access": "transductive if held-out target images are included"}
    write_json(a.output, report)
    print(report)
