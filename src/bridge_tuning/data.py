"""Manifest handling and preprocessing faithful to the segmentation pipeline."""
import csv
from functools import partial
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai import transforms as T


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


def binarize(x, mode="positive", values=(1,), threshold=0.5):
    if mode == "probability":
        binary = x >= threshold
    elif mode == "values":
        binary = torch.zeros_like(x, dtype=torch.bool)
        for value in values:
            binary |= x == value
    else:
        binary = x > 0
    return binary.to(dtype=torch.float32)


def preprocessing(config, with_label=True, training=False):
    p = config["preprocessing"]
    keys = ["image", "label"] if with_label else ["image"]
    modes = ("bilinear", "nearest") if with_label else "bilinear"
    operations = [T.LoadImaged(keys=keys), T.EnsureChannelFirstd(keys=keys),
                  T.Orientationd(keys=keys, axcodes="RAS"),
                  T.Spacingd(keys=keys, pixdim=tuple(p["spacing"]), mode=modes)]
    if p["modality"] == "ct":
        operations.append(T.ScaleIntensityRanged(keys="image", a_min=p["hu_window"][0],
            a_max=p["hu_window"][1], b_min=0.0, b_max=1.0, clip=True))
    else:
        operations.append(T.ScaleIntensityRangePercentilesd(keys="image", lower=p["percentiles"][0],
            upper=p["percentiles"][1], b_min=0.0, b_max=1.0, clip=True))
    operations.append(T.CropForegroundd(keys=keys, source_key="image", allow_smaller=True))
    if with_label:
        operations.append(T.Lambdad(keys="label", func=partial(binarize, mode=p["label_mode"],
            values=p["label_values"], threshold=p["label_threshold"])))
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
    if p["modality"] == "ct":
        operations += [T.RandScaleIntensityd(keys="image", factors=0.1, prob=p["intensity_probability"]),
                       T.RandShiftIntensityd(keys="image", offsets=0.1, prob=p["intensity_probability"])]
    return T.Compose(operations)
