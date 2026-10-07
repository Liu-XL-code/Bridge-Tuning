import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from bridge_tuning.splits import make_folds
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Create fixed subject-level test folds and seeded K-shot support sets")
    p.add_argument("--csv", "--manifest", dest="csv", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--shots", "--k", dest="shots", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--split-seed", type=int, default=1024)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if len(set(a.seeds)) != len(a.seeds):
        raise ValueError("Support seeds must be distinct")
    make_folds(a.csv, a.folds, a.shots, a.seeds, a.split_seed, a.output)
