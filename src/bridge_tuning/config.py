"""Validated configuration shared by training, validation, testing and inference."""
import copy
import json
from pathlib import Path

import yaml

DEFAULT = {
    "model": {"feature_size": 48, "in_channels": 1, "out_channels": 2,
              "strategy": "lora", "use_checkpoint": True, "lora_rank": 8,
              "lora_alpha": 16, "lora_dropout": 0.1, "lora_targets": ["qkv"],
              "freeze_encoder_convolutions": True},
    "preprocessing": {"modality": "ct", "orientation": "RAS",
                      "spacing": [1.5, 1.5, 2.0], "roi": [96, 96, 96],
                      "hu_window": [-175, 250], "percentiles": [0.5, 99.5],
                      "label_mode": "positive", "label_values": [1], "label_threshold": 0.5,
                      "flip_probability": 0.3, "rotate_probability": 0.3,
                      "intensity_probability": 0.2, "patches_per_volume": 2,
                      "geometry_tolerance_mm": 0.001},
    "training": {"epochs": 300, "batch_volumes": 4, "learning_rate": 0.0005,
                 "weight_decay": 0.00001, "warmup_epochs": 20, "seed": 0,
                 "schedule": "cosine", "decay_bias_and_norm": False,
                 "microbatch_patches": 2,
                 "workers": 0, "precision": "fp32", "selection": "final",
                 "validation_interval": 10, "early_stop": True,
                 "loss_smoothing_window": 10, "early_stop_patience": 30,
                 "early_stop_relative_threshold": 0.01},
    "evaluation": {"overlap": 0.5, "sw_batch_size": 1, "hd95_unit": "voxel",
                   "empty_mask_hd95": 100.0},
}


def merge(base, override):
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def validate(cfg):
    m, p, t, e = (cfg[k] for k in ("model", "preprocessing", "training", "evaluation"))
    if m["strategy"] not in {"fft", "lora", "bitfit", "linear_prob"}:
        raise ValueError("strategy must be fft, lora, bitfit or linear_prob")
    if m["in_channels"] != 1 or m["out_channels"] != 2:
        raise ValueError("This release implements one-channel, binary 3D segmentation")
    if m["feature_size"] <= 0 or m["feature_size"] % 12:
        raise ValueError("Swin UNETR feature_size must be a positive multiple of 12")
    if len(p["roi"]) != 3 or any(n < 64 or n % 32 for n in p["roi"]):
        raise ValueError("ROI axes must be multiples of 32 and >=64 (instance-normalization constraint)")
    if len(p["spacing"]) != 3 or any(s <= 0 for s in p["spacing"]):
        raise ValueError("Three positive voxel spacings are required")
    if p["modality"] not in {"ct", "mri"} or p["orientation"] != "RAS":
        raise ValueError("Supported preprocessing: CT/MRI with RAS orientation")
    if p["label_mode"] not in {"positive", "values", "probability"}:
        raise ValueError("label_mode must be positive, values or probability")
    if not p["label_values"] or not 0 <= p["label_threshold"] <= 1:
        raise ValueError("Require nonempty label_values and a probability threshold in [0,1]")
    if p["hu_window"][0] >= p["hu_window"][1]:
        raise ValueError("Invalid HU window")
    if not 0 <= p["percentiles"][0] < p["percentiles"][1] <= 100:
        raise ValueError("Invalid MRI percentile bounds")
    for k in ("flip_probability", "rotate_probability", "intensity_probability"):
        if not 0 <= p[k] <= 1:
            raise ValueError(f"{k} must be in [0,1]")
    if p["patches_per_volume"] < 1 or t["batch_volumes"] < 1 or t["epochs"] < 1:
        raise ValueError("Training sizes must be positive")
    if t["microbatch_patches"] < 1:
        raise ValueError("microbatch_patches must be positive")
    if t["workers"] < 0 or not 0 <= t["warmup_epochs"] < t["epochs"]:
        raise ValueError("Require nonnegative workers and 0 <= warmup_epochs < epochs")
    if t["schedule"] not in {"cosine", "constant"} or (t["schedule"] == "constant" and t["warmup_epochs"]):
        raise ValueError("schedule is cosine or constant; constant requires warmup_epochs=0")
    if t["learning_rate"] <= 0 or t["weight_decay"] < 0:
        raise ValueError("Learning rate must be positive and weight decay nonnegative")
    if t["precision"] not in {"fp32", "amp"} or t["selection"] not in {"final", "val_dice"}:
        raise ValueError("Unsupported precision or checkpoint selection")
    if t["validation_interval"] < 1 or t["loss_smoothing_window"] < 1 or t["early_stop_patience"] < 2:
        raise ValueError("Invalid validation or early-stopping intervals")
    if not 0 <= e["overlap"] < 1 or e["sw_batch_size"] < 1 or e["hd95_unit"] not in {"voxel", "mm"}:
        raise ValueError("Invalid evaluation settings")
    if m["lora_rank"] < 1 or not set(m["lora_targets"]).issubset({"qkv", "proj"}):
        raise ValueError("Invalid attention LoRA settings")
    return cfg


def load_config(path):
    with Path(path).open() as f:
        override = yaml.safe_load(f) or {}
    if set(override) - set(DEFAULT):
        raise ValueError(f"Unknown configuration sections: {set(override)-set(DEFAULT)}")
    for section, values in override.items():
        if not isinstance(values, dict) or set(values) - set(DEFAULT[section]):
            raise ValueError(f"Unknown or invalid settings in {section}")
    return validate(merge(DEFAULT, override))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
