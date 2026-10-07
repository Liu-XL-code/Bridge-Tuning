"""Command-line entry points; each command has a single explicit purpose."""
import argparse
import copy
import json
from pathlib import Path

import torch

from .config import load_config, validate, write_json
from .data import read_manifest
from .engine import evaluate, infer, select_device, train
from .model import load_model


def common_device(parser):
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=4)


def runtime(args):
    if args.cpu_threads < 1:
        raise ValueError("cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    return select_device(args.device)


def train_main():
    parser = argparse.ArgumentParser(description="Train a bridge model or adapt it on target support volumes")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", help="Optional, disjoint validation volumes; never a test manifest")
    parser.add_argument("--init", help="Release checkpoint for initialization")
    parser.add_argument("--reset-head", action="store_true", help="Reset the binary segmentation head before adaptation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--role", choices=["bridge", "target"], default="target")
    parser.add_argument("--seed", type=int)
    common_device(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    receipt = train(config, read_manifest(args.train_csv),
                    read_manifest(args.val_csv) if args.val_csv else [], runtime(args),
                    args.output, args.init, args.reset_head, args.role)
    print(json.dumps(receipt, indent=2))


def evaluate_main():
    parser = argparse.ArgumentParser(description="Validate or test a fixed checkpoint on a held-out manifest")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--hd95-unit", choices=["voxel", "mm"])
    parser.add_argument("--partition", choices=["validation", "test"], default="test")
    common_device(parser)
    args = parser.parse_args()
    device = runtime(args)
    model, cfg, ck = load_model(args.checkpoint, device)
    cfg = copy.deepcopy(cfg)
    if args.hd95_unit:
        cfg["evaluation"]["hd95_unit"] = args.hd95_unit
    report = evaluate(model, read_manifest(args.csv), validate(cfg), device,
                      args.output, args.save_predictions)
    report["partition"] = args.partition
    report["checkpoint_epoch"] = ck["epoch"]
    report["checkpoint_role"] = ck["role"]
    write_json(Path(args.output) / "metrics.json", report)
    print(json.dumps(report["summary"], indent=2))


def infer_main():
    parser = argparse.ArgumentParser(description="Infer a native-space binary mask from a 3D CT/MRI volume")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True, help="Output NIfTI path (.nii.gz)")
    common_device(parser)
    args = parser.parse_args()
    device = runtime(args)
    model, cfg, _ = load_model(args.checkpoint, device)
    print(json.dumps(infer(model, args.image, cfg, device, args.output), indent=2))
