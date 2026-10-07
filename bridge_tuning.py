#!/usr/bin/env python3
"""Bridge-Tuning CT pipeline: foundation A, full bridge B, few-shot LoRA C.

All preprocessing, training, held-out evaluation and native-grid inference live here.
MIT license; copyright (c) 2026 Bridge-Tuning authors. Full notice in README.md.
"""
import argparse
import copy
import json
from pathlib import Path
import yaml
import csv
import nibabel as nib
import numpy as np
import torch
from monai import transforms as T
from scipy.ndimage import binary_erosion, distance_transform_edt
import inspect
from monai.networks.nets import SwinUNETR
from torch import nn
import math
import random
import shutil
import time
from monai.data import DataLoader, Dataset, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.utils import set_determinism
from scipy.spatial.distance import cdist


# CONFIG

DEFAULT = {
    "model": {"feature_size": 48, "in_channels": 1, "out_channels": 2,
              "strategy": "lora", "use_checkpoint": True, "lora_rank": 8,
              "lora_alpha": 16, "lora_dropout": 0.1, "lora_targets": ["qkv"],
              "freeze_encoder_convolutions": True},
    "preprocessing": {"orientation": "RAS",
                      "spacing": [1.5, 1.5, 2.0], "roi": [96, 96, 96],
                      "hu_window": [-175, 250], "flip_probability": 0.3, "rotate_probability": 0.3,
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
    if m["strategy"] not in {"fft", "lora"}:
        raise ValueError("strategy must be fft (bridge) or lora (target)")
    if m["in_channels"] != 1 or m["out_channels"] != 2:
        raise ValueError("This release implements one-channel, binary 3D segmentation")
    if m["feature_size"] <= 0 or m["feature_size"] % 12:
        raise ValueError("Swin UNETR feature_size must be a positive multiple of 12")
    if len(p["roi"]) != 3 or any(n < 64 or n % 32 for n in p["roi"]):
        raise ValueError("ROI axes must be multiples of 32 and >=64 (instance-normalization constraint)")
    if len(p["spacing"]) != 3 or any(s <= 0 for s in p["spacing"]):
        raise ValueError("Three positive voxel spacings are required")
    if p["orientation"] != "RAS":
        raise ValueError("CT preprocessing requires RAS orientation")
    if p["hu_window"][0] >= p["hu_window"][1]:
        raise ValueError("Invalid HU window")
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


def load_config(path, stage="C"):
    """Read one shared CT recipe and resolve exactly one A/B/C stage."""
    if stage not in {"A", "B", "C"}:
        raise ValueError("Stage must be A, B or C")
    with Path(path).open() as f:
        values = yaml.safe_load(f) or {}
    stages = values.pop("stages", {})
    if set(stages) - {"A", "B", "C"}:
        raise ValueError("Unknown stage in configuration")
    stage_values = stages.get(stage, {})
    for override in [values, stage_values]:
        if not isinstance(override, dict) or set(override) - set(DEFAULT):
            raise ValueError("Unknown configuration section")
        for section, settings in override.items():
            if not isinstance(settings, dict) or set(settings) - set(DEFAULT[section]):
                raise ValueError(f"Unknown or invalid settings in {section}")
    cfg = validate(merge(merge(DEFAULT, values), stage_values))
    expected = "lora" if stage == "C" else "fft"
    if cfg["model"]["strategy"] != expected:
        raise ValueError(f"Stage {stage} requires strategy {expected}")
    return cfg


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


# DATA

def read_manifest(csv_path, labels_required=True):
    csv_path = Path(csv_path).resolve()
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    required = {"id", "image"} | ({"label"} if labels_required else set())
    if not required.issubset(reader.fieldnames or []):
        raise ValueError(f"Manifest needs columns {sorted(required)}")
    items = []
    for row in rows:
        case = {"id": row["id"].strip(), "subject_id": row.get("subject_id", row["id"]).strip()}
        if not case["id"] or not case["subject_id"]:
            raise ValueError("Empty case or subject identifier")
        if any(c in case["id"] for c in ("/", "\\", "\x00")) or case["id"] in {".", ".."}:
            raise ValueError("Case ID must be a filename-safe identifier")
        for key in (("image", "label") if labels_required else ("image",)):
            if row.get(key, "").strip():
                p = Path(row[key].strip())
                case[key] = str((csv_path.parent / p).resolve()) if not p.is_absolute() else str(p.resolve())
                if not Path(case[key]).is_file():
                    raise FileNotFoundError(f"Missing {key} for case {case['id']}")
            elif key == "image" or labels_required:
                raise ValueError(f"Missing {key} for case {case['id']}")
        items.append(case)
    if not items or len({x["id"] for x in items}) != len(items) or len({x["image"] for x in items}) != len(items):
        raise ValueError("Manifest must be nonempty with unique case IDs and image files")
    return items


def assert_disjoint(*partitions):
    for i, left in enumerate(partitions):
        for right in partitions[i + 1:]:
            if {x["image"] for x in left} & {x["image"] for x in right}:
                raise ValueError("Image overlap between partitions")
            if {x["subject_id"] for x in left} & {x["subject_id"] for x in right}:
                raise ValueError("Subject overlap between partitions (including repeated time frames)")


def check_geometry(items, tolerance=0.001):
    for item in items:
        image = nib.load(item["image"])
        if len(image.shape) != 3:
            raise ValueError(f"Case {item['id']}: expected a static 3D volume, got {image.shape}")
        if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) <= 0:
            raise ValueError(f"Case {item['id']}: invalid image affine")
        if image.header.get_xyzt_units()[0] not in {"mm", "unknown"}:
            raise ValueError(f"Case {item['id']}: convert declared spatial units to mm first")
        if "label" in item:
            label = nib.load(item["label"])
            if image.shape != label.shape or not np.allclose(image.affine, label.affine, rtol=0, atol=tolerance):
                raise ValueError(f"Case {item['id']}: image/label grids differ; verify alignment before training. Source files were not modified.")
            if label.header.get_xyzt_units()[0] not in {"mm", "unknown"}:
                raise ValueError(f"Case {item['id']}: label spatial units must be mm")


def binarize(x):
    """Private CT labels: all positive tumor labels are foreground."""
    return (x > 0).to(dtype=torch.float32)


def preprocessing(config, with_label=True, training=False):
    p = config["preprocessing"]
    keys = ["image", "label"] if with_label else ["image"]
    modes = ("bilinear", "nearest") if with_label else "bilinear"
    operations = [T.LoadImaged(keys=keys), T.EnsureChannelFirstd(keys=keys),
                  T.Orientationd(keys=keys, axcodes="RAS"),
                  T.Spacingd(keys=keys, pixdim=tuple(p["spacing"]), mode=modes)]
    operations.append(T.ScaleIntensityRanged(keys="image", a_min=p["hu_window"][0],
        a_max=p["hu_window"][1], b_min=0.0, b_max=1.0, clip=True))
    operations.append(T.CropForegroundd(keys=keys, source_key="image", allow_smaller=True))
    if with_label:
        operations.append(T.Lambdad(keys="label", func=binarize))
    if training:
        operations.append(T.SpatialPadd(keys=keys, spatial_size=tuple(p["roi"])))
    operations.append(T.EnsureTyped(keys=keys, dtype=torch.float32, track_meta=True))
    return T.Compose(operations)


def augmentation(config):
    p = config["preprocessing"]
    operations = [T.RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
        spatial_size=tuple(p["roi"]), pos=1, neg=1, num_samples=p["patches_per_volume"],
        image_key="image", image_threshold=0)]
    operations += [T.RandFlipd(keys=["image", "label"], prob=p["flip_probability"], spatial_axis=a) for a in range(3)]
    operations += [T.RandRotate90d(keys=["image", "label"], prob=p["rotate_probability"], max_k=3)]
    operations += [T.RandScaleIntensityd(keys="image", factors=0.1, prob=p["intensity_probability"]),
                       T.RandShiftIntensityd(keys="image", offsets=0.1, prob=p["intensity_probability"])]
    return T.Compose(operations)


# METRICS

def binary_metrics(prediction, target, spacing=(1, 1, 1), empty_hd95=100.0):
    pred, gt = np.asarray(prediction) > 0, np.asarray(target) > 0
    if pred.shape != gt.shape or pred.ndim != 3:
        raise ValueError("Metrics require matching 3D binary volumes")
    intersection = int(np.count_nonzero(pred & gt))
    size = int(pred.sum()) + int(gt.sum())
    dice = float((2 * intersection + 1e-5) / (size + 1e-5))
    if not pred.any() and not gt.any():
        hd95 = 0.0
    elif not pred.any() or not gt.any():
        hd95 = float(empty_hd95)
    else:
        ps = pred & ~binary_erosion(pred)
        gs = gt & ~binary_erosion(gt)
        p_to_g = distance_transform_edt(~gs, sampling=spacing)[ps]
        g_to_p = distance_transform_edt(~ps, sampling=spacing)[gs]
        hd95 = float(np.percentile(np.concatenate([p_to_g, g_to_p]), 95))
    return {"dice": dice, "hd95": hd95, "prediction_empty": not bool(pred.any()),
            "ground_truth_empty": not bool(gt.any())}


def summarize(cases):
    if not cases:
        raise ValueError("Cannot aggregate empty results")
    result = {"n_cases": len(cases), "aggregation_unit": "case", "std_definition": "sample, ddof=1; null when n<2"}
    for metric in ("dice", "hd95"):
        values = np.asarray([c[metric] for c in cases], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite metrics")
        result[metric + "_mean"] = float(values.mean())
        result[metric + "_sample_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
    return result


# MODEL

class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank, alpha, dropout):
        super().__init__()
        self.original_layer = original_layer
        for parameter in original_layer.parameters():
            parameter.requires_grad_(False)
        self.scaling = alpha / rank
        self.lora_a = nn.Linear(original_layer.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, original_layer.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        return self.original_layer(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


def is_encoder(name, freeze_convolutions=True):
    top = name.split(".")[0]
    return top == "swinViT" or (freeze_convolutions and top.startswith("encoder"))


def apply_strategy(model, cfg):
    mode = cfg["strategy"]
    if mode not in {"fft", "lora"}:
        raise ValueError("Only bridge FFT and target LoRA are implemented")
    freeze_convs = cfg["freeze_encoder_convolutions"]
    if mode == "lora":
        for name, module in list(model.named_modules()):
            if not name.startswith("swinViT"):
                continue
            for child_name, child in list(module.named_children()):
                if child_name in cfg["lora_targets"] and isinstance(child, nn.Linear):
                    setattr(module, child_name, LoRALinear(child, cfg["lora_rank"],
                            cfg["lora_alpha"], cfg["lora_dropout"]))
    for name, parameter in model.named_parameters():
        encoder = is_encoder(name, freeze_convs)
        if mode == "fft":
            trainable = True
        else:
            trainable = not encoder or ".lora_a." in name or ".lora_b." in name
        parameter.requires_grad_(trainable)
    return model


def build_model(cfg, apply_peft=True):
    m = cfg["model"]
    kwargs = dict(in_channels=m["in_channels"], out_channels=m["out_channels"],
                  feature_size=m["feature_size"], use_checkpoint=m["use_checkpoint"], spatial_dims=3)
    if "img_size" in inspect.signature(SwinUNETR).parameters:
        kwargs["img_size"] = tuple(cfg["preprocessing"]["roi"])
    model = SwinUNETR(**kwargs)
    return apply_strategy(model, m) if apply_peft else model


def read_checkpoint(path):
    # Distributed packages contain tensors and built-in Python values only.
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or "state_dict" not in value or value.get("schema_version") != 1:
        raise ValueError("Expected a Bridge-Tuning release checkpoint; use the init command for a trusted legacy file")
    if not all(isinstance(v, torch.Tensor) for v in value["state_dict"].values()):
        raise ValueError("Invalid checkpoint state_dict")
    return value


def merged_state(model):
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
             if ".lora_a." not in k and ".lora_b." not in k and ".original_layer." not in k}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[name + ".weight"] = (module.original_layer.weight +
                module.scaling * (module.lora_b.weight @ module.lora_a.weight)).detach().cpu().clone()
            if module.original_layer.bias is not None:
                state[name + ".bias"] = module.original_layer.bias.detach().cpu().clone()
    return state


def initialize(model, checkpoint, reset_head=False):
    ck = read_checkpoint(checkpoint)
    state = ck["state_dict"]
    if any(".lora_a." in k for k in state):
        source = build_model(ck["config"])
        source.load_state_dict(state, strict=True)
        state = merged_state(source)
    destination = model.state_dict()
    matched, ignored = {}, []
    for key, value in state.items():
        if key.startswith("out.") and (reset_head or key not in destination or destination[key].shape != value.shape):
            ignored.append(key)
        elif key in destination and destination[key].shape == value.shape:
            matched[key] = value
        else:
            raise ValueError(f"Unexpected non-head tensor during initialization: {key}")
    missing = [k for k in destination if k not in matched]
    if any(not k.startswith("out.") for k in missing):
        raise ValueError(f"Initialization misses non-head tensors: {missing}")
    if not matched:
        raise ValueError("No usable pretrained tensors")
    model.load_state_dict(matched, strict=False)
    return {"loaded_tensors": len(matched), "reset_or_missing_head_tensors": missing,
            "checkpoint_role": ck.get("role", "unspecified")}


def save_checkpoint(path, model, config, epoch, role, extra=None):
    path = Path(path)
    payload = {"schema_version": 1, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "config": config, "epoch": epoch, "role": role, "metadata": extra or {}}
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_model(path, device):
    checkpoint = read_checkpoint(path)
    if checkpoint.get("metadata", {}).get("initialization_only"):
        raise ValueError("This is an initialization-only backbone; train a segmentation model before inference")
    validate(checkpoint["config"])
    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint["config"], checkpoint


# ENGINE

def select_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")
    return torch.device(value)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)


def learning_rate(epoch, training):
    if training["schedule"] == "constant":
        return training["learning_rate"]
    warmup, total, base = (training[k] for k in ("warmup_epochs", "epochs", "learning_rate"))
    if warmup and epoch < warmup:
        return base * (epoch + 1) / warmup
    progress = (epoch - warmup) / (total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * progress))


def should_stop(losses, cfg):
    window, patience = cfg["loss_smoothing_window"], cfg["early_stop_patience"]
    if not cfg["early_stop"] or len(losses) < window + patience:
        return False
    smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
    return bool(smoothed[-patience] - smoothed[-1] < cfg["early_stop_relative_threshold"] * smoothed[-patience])


def optimizer_for(model, training):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if not training["decay_bias_and_norm"] and
                (parameter.ndim == 1 or name.endswith(".bias")) else decay).append(parameter)
    return torch.optim.AdamW([{"params": decay, "weight_decay": training["weight_decay"]},
                            {"params": no_decay, "weight_decay": 0.0}], lr=training["learning_rate"])


@torch.inference_mode()
def predict_tensor(model, image, cfg, device):
    # Window blending matches the original default constant blend with 50% overlap.
    with torch.autocast(device_type=device.type, enabled=cfg["training"]["precision"] == "amp" and device.type == "cuda"):
        logits = sliding_window_inference(image.unsqueeze(0).to(device),
            tuple(cfg["preprocessing"]["roi"]), cfg["evaluation"]["sw_batch_size"], model,
            overlap=cfg["evaluation"]["overlap"], mode="constant")
    return logits.float().argmax(dim=1)[0].cpu()


@torch.inference_mode()
def evaluate(model, items, cfg, device, output=None, save_predictions=False):
    check_geometry(items, cfg["preprocessing"]["geometry_tolerance_mm"])
    model.eval()
    transform = preprocessing(cfg, with_label=True)
    cases = []
    unit = cfg["evaluation"]["hd95_unit"]
    if output:
        Path(output).mkdir(parents=True, exist_ok=True)
    for item in items:
        data = transform(item)
        pred = predict_tensor(model, data["image"], cfg, device)
        gt = data["label"][0].as_tensor().cpu().numpy()
        spacing = tuple(np.linalg.norm(data["image"].affine[:3, :3].cpu().numpy(), axis=0)) if unit == "mm" else (1, 1, 1)
        metrics = binary_metrics(pred.numpy(), gt, spacing, cfg["evaluation"]["empty_mask_hd95"])
        cases.append({"id": item["id"], **metrics})
        if output and save_predictions:
            save_native_prediction(pred, data, transform, item["image"], Path(output) / (item["id"] + "_pred.nii.gz"))
    report = {"evaluation_space": "preprocessed RAS/spaced/foreground-cropped grid", "hd95_unit": unit,
              "empty_mask_hd95": cfg["evaluation"]["empty_mask_hd95"], "no_low_dice_exclusion": True,
              "cases": cases, "summary": summarize(cases)}
    if output:
        write_json(Path(output) / "metrics.json", report)
    return report


def save_native_prediction(prediction, data, transform, image_path, output_path):
    source = dict(data)
    source["pred"] = prediction.unsqueeze(0)
    inverted = T.Invertd(keys="pred", transform=transform, orig_keys="image",
                         nearest_interp=True, to_tensor=True)(source)["pred"]
    result = inverted[0].detach().cpu().numpy().astype(np.uint8)
    original = nib.load(image_path)
    if result.shape != original.shape:
        raise RuntimeError(f"Inverse transform did not restore native shape: {result.shape} vs {original.shape}")
    header = original.header.copy()
    header.set_data_dtype(np.uint8)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(result, original.affine, header), str(output_path))


def infer(model, image_path, cfg, device, output_path):
    item = {"id": "input", "image": str(Path(image_path).resolve())}
    check_geometry([item], cfg["preprocessing"]["geometry_tolerance_mm"])
    transform = preprocessing(cfg, with_label=False)
    data = transform(item)
    pred = predict_tensor(model, data["image"], cfg, device)
    save_native_prediction(pred, data, transform, item["image"], output_path)
    return {"prediction_path": str(output_path), "space": "original input NIfTI grid",
            "foreground_voxels_preprocessed_grid": int(pred.sum())}


def train(config, train_items, validation_items, device, output, initial=None, reset_head=False, role="target"):
    cfg = copy.deepcopy(config)
    output = Path(output)
    if (output / "final.pt").exists() or (output / "history.json").exists():
        raise FileExistsError("Training output already exists; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    assert_disjoint(train_items, validation_items)
    check_geometry(train_items + validation_items, cfg["preprocessing"]["geometry_tolerance_mm"])
    t, p = cfg["training"], cfg["preprocessing"]
    if t["selection"] == "val_dice" and not validation_items:
        raise ValueError("val_dice selection requires a separate validation manifest")
    seed_everything(t["seed"])
    model = build_model(cfg, apply_peft=False)
    initialization = initialize(model, initial, reset_head) if initial else {"random_initialization": True}
    apply_strategy(model, cfg["model"])
    model = model.to(device)
    optimizer = optimizer_for(model, t)
    amp = t["precision"] == "amp" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    operations = T.Compose([preprocessing(cfg, training=True), augmentation(cfg)])
    operations.set_random_state(seed=t["seed"])
    dataset = Dataset(train_items, transform=operations)
    loader = DataLoader(dataset, batch_size=t["batch_volumes"], shuffle=True,
        num_workers=t["workers"], collate_fn=list_data_collate, drop_last=False)
    total_parameters = sum(x.numel() for x in model.parameters())
    trainable_parameters = sum(x.numel() for x in model.parameters() if x.requires_grad)
    history, losses, best, best_epoch = [], [], -math.inf, None
    started = time.time()
    write_json(output / "config.json", cfg)
    write_json(output / "split_ids.json", {"train": [x["id"] for x in train_items],
        "validation": [x["id"] for x in validation_items], "test_accessed": False})
    for epoch in range(t["epochs"]):
        model.train()
        lr = learning_rate(epoch, t)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch_losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            # Preserve the effective patch batch with weighted accumulation, including
            # a smaller final microbatch. DiceCELoss uses mean reduction per example.
            patch_count, weighted_loss = len(batch["image"]), 0.0
            for start in range(0, patch_count, t["microbatch_patches"]):
                stop = min(start + t["microbatch_patches"], patch_count)
                fraction = (stop-start) / patch_count
                with torch.autocast(device_type=device.type, enabled=amp):
                    loss = loss_function(model(batch["image"][start:stop].to(device)),
                                         batch["label"][start:stop].to(device))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                scaler.scale(loss * fraction).backward()
                weighted_loss += float(loss.detach().cpu()) * fraction
            scaler.step(optimizer)
            scaler.update()
            batch_losses.append(weighted_loss)
        if not batch_losses:
            raise RuntimeError("Training loader yielded zero batches")
        losses.append(float(np.mean(batch_losses)))
        record = {"epoch": epoch + 1, "loss": losses[-1], "learning_rate": lr}
        if validation_items and ((epoch + 1) % t["validation_interval"] == 0 or epoch == t["epochs"] - 1):
            validation = evaluate(model, validation_items, cfg, device)
            record["validation"] = validation["summary"]
            metric = validation["summary"]["dice_mean"]
            if metric > best:
                best, best_epoch = metric, epoch + 1
                if t["selection"] == "val_dice":
                    save_checkpoint(output / "best.pt", model, cfg, epoch + 1, role, {"validation_dice": best})
        history.append(record)
        write_json(output / "history.json", history)
        print(f"epoch {epoch+1}/{t['epochs']} loss={losses[-1]:.6f} lr={lr:.6g}", flush=True)
        if should_stop(losses, t):
            break
    # Include final validation even if loss early stopping occurs between validation intervals.
    if validation_items and "validation" not in history[-1]:
        final_validation = evaluate(model, validation_items, cfg, device)
        history[-1]["validation"] = final_validation["summary"]
        if final_validation["summary"]["dice_mean"] > best:
            best, best_epoch = final_validation["summary"]["dice_mean"], len(history)
            if t["selection"] == "val_dice":
                save_checkpoint(output / "best.pt", model, cfg, len(history), role, {"validation_dice": best})
        write_json(output / "history.json", history)
    save_checkpoint(output / "final.pt", model, cfg, len(history), role)
    chosen = "best.pt" if t["selection"] == "val_dice" else "final.pt"
    shutil.copyfile(output / chosen, output / "selected.pt")
    receipt = {"epochs_trained": len(history), "early_stopped": len(history) < t["epochs"],
        "selection": t["selection"], "selected_checkpoint": "selected.pt", "selected_source": chosen,
        "best_validation_epoch": best_epoch, "training_seed": t["seed"], "train_volumes": len(train_items),
        "validation_volumes": len(validation_items), "patches_per_volume": p["patches_per_volume"],
        "maximum_patch_batch_size": t["batch_volumes"] * p["patches_per_volume"],
        "microbatch_patches": t["microbatch_patches"],
        "parameters": total_parameters, "trainable_parameters": trainable_parameters,
        "initialization": initialization, "test_accessed": False, "seconds": round(time.time() - started, 3)}
    write_json(output / "training_receipt.json", receipt)
    return receipt


# SPLITS

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
            # Keep one support volume per sampled subject when repeat scans exist.
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


# SELECTION

def features(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or len(value) < 2 or not np.isfinite(value).all():
        raise ValueError("Features must be a finite [N,D] array with N>=2")
    return value


def coverage(value):
    value = features(value)
    return float(np.square(value - value.mean(axis=0)).sum() / (len(value) - 1))


def assess(target, bridge, batch_size=128):
    target, bridge = features(target), features(bridge)
    if target.shape[1] != bridge.shape[1] or batch_size < 1:
        raise ValueError("Feature dimensions must match and batch size must be positive")
    nearest = [cdist(target[i:i+batch_size], bridge).min(axis=1)
               for i in range(0, len(target), batch_size)]
    return {"D_target_to_bridge": float(np.concatenate(nearest).mean()),
            "W_bridge": coverage(bridge), "W_target": coverage(target),
            "delta_W": coverage(bridge)-coverage(target),
            "n_bridge_images": len(bridge), "n_target_images": len(target)}

# FOUNDATION INITIALIZATION, FROZEN BRIDGE ASSESSMENT AND LOCAL OVERLAYS

def convert_foundation(source, config, destination, trusted):
    """Import a locally obtained trusted full Swin UNETR initializer; omit its head."""
    if not trusted:
        raise ValueError("init requires --trusted for a locally trusted legacy checkpoint")
    ck = torch.load(source, map_location="cpu", weights_only=False)
    raw = ck.get("state_dict", ck.get("model", ck)) if isinstance(ck, dict) else ck
    if not isinstance(raw, dict):
        raise ValueError("Foundation file does not contain a state dictionary")
    state = {}
    for key, value in raw.items():
        if not isinstance(value, torch.Tensor):
            continue
        while key.startswith("module.") or key.startswith("model."):
            key = key.split(".", 1)[1]
        if not key.startswith(("out.", "classifier.")):
            state[key] = value.detach().cpu().contiguous()
    if not state:
        raise ValueError("No foundation tensors found")
    cfg = load_config(config, "A")
    # Validate compatibility before saving, rather than silently accepting missing layers.
    reference = build_model(cfg, apply_peft=False).state_dict()
    required = {k for k in reference if not k.startswith("out.")}
    if set(state) != required or any(state[k].shape != reference[k].shape for k in required):
        raise ValueError("Foundation must match the configured full Swin UNETR backbone and decoder")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": 1, "state_dict": state, "config": cfg,
        "role": "foundation", "epoch": 0,
        "metadata": {"initialization_only": True, "upstream_head_removed": True}}, destination)
    return {"role": "foundation", "tensors": len(state), "initialization_only": True}


def extract_features(model, cfg, manifest, device):
    """Image-only stage-3 pooling; even label paths are discarded before I/O."""
    items = read_manifest(manifest, labels_required=False)
    check_geometry(items, cfg["preprocessing"]["geometry_tolerance_mm"])
    transform = T.Compose([preprocessing(cfg, with_label=False),
        T.Resized(keys="image", spatial_size=tuple(cfg["preprocessing"]["roi"]), mode="area")])
    cache, rows = {}, []
    def hook(module, inputs, output):
        cache["features"] = output.detach().mean(dim=(2, 3, 4)).cpu().numpy()
    handle = model.swinViT.layers3[-1].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            for item in items:
                cache.clear()
                model.swinViT(transform(item)["image"].unsqueeze(0).to(device))
                rows.append(cache["features"][0])
    finally:
        handle.remove()
    return np.stack(rows)


def render_overlay(image_path, prediction_path, output, label_path=None):
    """Create a local native-grid input/(optional GT)/prediction montage."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    image = nib.load(image_path)
    volumes = [nib.load(prediction_path)]
    if label_path:
        volumes.append(nib.load(label_path))
    if any(v.shape != image.shape or not np.allclose(v.affine, image.affine, atol=0.001, rtol=0) for v in volumes):
        raise ValueError("Overlay inputs must share the same native grid")
    ct = np.asarray(image.dataobj)
    prediction = np.asarray(volumes[0].dataobj) > 0
    ground_truth = np.asarray(volumes[1].dataobj) > 0 if label_path else None
    focus = ground_truth if ground_truth is not None else prediction
    center = int(focus.sum(axis=(0, 1)).argmax()) if focus.any() else ct.shape[2] // 2
    planes = sorted({max(0, min(ct.shape[2] - 1, center + offset)) for offset in [-4, 0, 4]})
    masks = [None] + ([ground_truth] if ground_truth is not None else []) + [prediction]
    titles = ["Input CT"] + (["Ground truth"] if ground_truth is not None else []) + ["Prediction"]
    figure, axes = plt.subplots(len(planes), len(masks), figsize=(3 * len(masks), 3 * len(planes)), squeeze=False)
    for row, z in enumerate(planes):
        for col, mask in enumerate(masks):
            ax = axes[row, col]
            ax.imshow(ct[:, :, z].T, cmap="gray", vmin=-175, vmax=250, origin="lower")
            if mask is not None:
                ax.imshow(np.ma.masked_where(~mask[:, :, z].T, mask[:, :, z].T),
                          cmap="autumn", alpha=0.5, vmin=0, vmax=1, origin="lower")
            ax.set_axis_off()
            if row == 0:
                ax.set_title(titles[col])
    figure.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


# SINGLE COMMAND-LINE ENTRY POINT

def main():
    parser = argparse.ArgumentParser(description="Bridge-Tuning: CT A -> B FFT -> C LoRA, assessment and inference")
    commands = parser.add_subparsers(dest="command", required=True)
    def device_options(command):
        command.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:0")
        command.add_argument("--cpu-threads", type=int, default=4)
    init = commands.add_parser("init", help="Import a separately obtained foundation initializer A")
    init.add_argument("--input", required=True)
    init.add_argument("--config", default="bridge_tuning.yaml")
    init.add_argument("--output", required=True)
    init.add_argument("--trusted", action="store_true")
    split = commands.add_parser("split", help="Fix subject-level test folds and sample K labeled support volumes")
    split.add_argument("--csv", required=True)
    split.add_argument("--folds", type=int, default=5)
    split.add_argument("--shots", type=int, default=5)
    split.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    split.add_argument("--split-seed", type=int, default=1024)
    split.add_argument("--output", required=True)
    fit = commands.add_parser("train", help="Train stage B with FFT or stage C with LoRA")
    fit.add_argument("--config", default="bridge_tuning.yaml")
    fit.add_argument("--stage", choices=["B", "C"], required=True)
    fit.add_argument("--train-csv", required=True)
    fit.add_argument("--val-csv", help="Optional disjoint validation manifest; never a test set")
    fit.add_argument("--init", required=True)
    fit.add_argument("--seed", type=int)
    fit.add_argument("--output", required=True)
    device_options(fit)
    score = commands.add_parser("evaluate", help="Validate/test a fixed selected checkpoint")
    score.add_argument("--checkpoint", required=True)
    score.add_argument("--csv", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--partition", choices=["validation", "test"], default="test")
    score.add_argument("--hd95-unit", choices=["voxel", "mm"])
    score.add_argument("--save-predictions", action="store_true")
    device_options(score)
    predict = commands.add_parser("infer", help="Predict a native-grid mask; optionally render a local montage")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--image", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--overlay", help="Optional output PNG path")
    predict.add_argument("--label", help="Optional local ground truth for the montage only")
    device_options(predict)
    criterion = commands.add_parser("assess", help="Compute D and delta W from frozen unlabeled B/C features")
    criterion.add_argument("--checkpoint", required=True)
    criterion.add_argument("--target-csv", required=True)
    criterion.add_argument("--bridge-csv", required=True)
    criterion.add_argument("--output", required=True)
    device_options(criterion)
    args = parser.parse_args()
    if getattr(args, "cpu_threads", 4) < 1:
        parser.error("cpu-threads must be positive")
    torch.set_num_threads(getattr(args, "cpu_threads", 4))
    if args.command == "init":
        result = convert_foundation(args.input, args.config, args.output, args.trusted)
    elif args.command == "split":
        rows = make_folds(args.csv, args.folds, args.shots, args.seeds, args.split_seed, args.output)
        result = {"folds": args.folds, "support_seeds": args.seeds, "registered_runs": len(rows)}
    elif args.command == "train":
        cfg = load_config(args.config, args.stage)
        if args.seed is not None:
            cfg["training"]["seed"] = args.seed
        expected = "foundation" if args.stage == "B" else "bridge"
        if read_checkpoint(args.init)["role"] != expected:
            raise ValueError(f"Stage {args.stage} requires a {expected} checkpoint")
        result = train(cfg, read_manifest(args.train_csv),
            read_manifest(args.val_csv) if args.val_csv else [], select_device(args.device),
            args.output, args.init, False, "bridge" if args.stage == "B" else "target")
    elif args.command == "evaluate":
        device = select_device(args.device)
        model, cfg, ck = load_model(args.checkpoint, device)
        cfg = copy.deepcopy(cfg)
        if args.hd95_unit:
            cfg["evaluation"]["hd95_unit"] = args.hd95_unit
        report = evaluate(model, read_manifest(args.csv), validate(cfg), device, args.output, args.save_predictions)
        report.update(partition=args.partition, checkpoint_epoch=ck["epoch"], checkpoint_role=ck["role"])
        write_json(Path(args.output) / "metrics.json", report)
        result = report["summary"]
    elif args.command == "infer":
        device = select_device(args.device)
        model, cfg, _ = load_model(args.checkpoint, device)
        result = infer(model, args.image, cfg, device, args.output)
        if args.overlay:
            render_overlay(args.image, args.output, args.overlay, args.label)
            result["overlay_path"] = args.overlay
    else:
        device = select_device(args.device)
        ck = read_checkpoint(args.checkpoint)
        if ck["role"] != "foundation":
            raise ValueError("Pre-adaptation assessment requires the fixed foundation checkpoint A")
        cfg = validate(ck["config"])
        model = build_model(cfg, apply_peft=False)
        initialize(model, args.checkpoint)
        model.requires_grad_(False).to(device).eval()
        result = assess(extract_features(model, cfg, args.target_csv, device),
                        extract_features(model, cfg, args.bridge_csv, device))
        result.update(labels_accessed=False, setting="transductive", feature_definition="Swin stage-3 global spatial average")
        write_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
