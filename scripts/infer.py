#!/usr/bin/env python
"""Single-volume inference for Bridge-Tuning checkpoints."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from monai import transforms as T
from monai.inferers import sliding_window_inference

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "lib"))

import lib.models as models  # noqa: E402


MODEL_REGISTRY = {
    "full": "SwinUNETRModel",
    "lora": "SwinUNETRLoRA",
    "bitfit": "SwinUNETRBitFit",
    "linear": "SwinUNETRLinear_Prob",
}


def build_transform(args):
    return T.Compose([
        T.LoadImaged(keys=["image"]),
        T.EnsureChannelFirstd(keys=["image"]),
        T.Orientationd(keys=["image"], axcodes="RAS"),
        T.Spacingd(keys=["image"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear",)),
        T.ScaleIntensityRanged(
            keys=["image"],
            a_min=args.a_min,
            a_max=args.a_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        T.CropForegroundd(keys=["image"], source_key="image"),
        T.ToTensord(keys=["image"]),
    ])


def normalize_state_dict(state):
    if "state_dict" in state:
        state = state["state_dict"]
    elif "model" in state:
        state = state["model"]

    out = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        out[key] = value
    return out


def load_model(args, device):
    class_name = MODEL_REGISTRY[args.model]
    model_class = getattr(models, class_name)
    model = model_class(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = normalize_state_dict(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in getattr(incompatible, "missing_keys", []) if "out.conv" not in k]
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing:
        print(f"Warning: {len(missing)} missing keys")
    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected keys")

    model.to(device)
    model.eval()
    return model


def save_nifti_if_possible(mask, out_path):
    try:
        import nibabel as nib
    except Exception:
        print("nibabel is not installed; skipping NIfTI export")
        return
    affine = np.eye(4, dtype=np.float32)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), str(out_path))


def save_overlay(image, mask, out_path):
    import matplotlib.pyplot as plt

    image = np.asarray(image)
    mask = np.asarray(mask)
    if mask.sum() > 0:
        z = int(mask.reshape(-1, mask.shape[-1]).sum(axis=0).argmax())
    else:
        z = mask.shape[-1] // 2

    img_slice = image[..., z]
    mask_slice = mask[..., z]

    plt.figure(figsize=(6, 6))
    plt.imshow(img_slice, cmap="gray")
    plt.imshow(np.ma.masked_where(mask_slice == 0, mask_slice), cmap="autumn", alpha=0.45)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), default="lora")
    parser.add_argument("--output-dir", default="outputs/inference")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--out-channels", type=int, default=2)
    parser.add_argument("--feature-size", type=int, default=48)
    parser.add_argument("--roi-x", type=int, default=96)
    parser.add_argument("--roi-y", type=int, default=96)
    parser.add_argument("--roi-z", type=int, default=96)
    parser.add_argument("--space-x", type=float, default=1.5)
    parser.add_argument("--space-y", type=float, default=1.5)
    parser.add_argument("--space-z", type=float, default=2.0)
    parser.add_argument("--a-min", type=float, default=-175.0)
    parser.add_argument("--a-max", type=float, default=250.0)
    parser.add_argument("--sw-batch-size", type=int, default=4)
    parser.add_argument("--overlap", type=float, default=0.5)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = load_model(args, device)

    data = build_transform(args)({"image": args.image})
    image = data["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = sliding_window_inference(
            image,
            roi_size=(args.roi_x, args.roi_y, args.roi_z),
            sw_batch_size=args.sw_batch_size,
            predictor=model,
            overlap=args.overlap,
        )
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    np.save(out_dir / "prediction_preprocessed.npy", pred)
    save_nifti_if_possible(pred, out_dir / "prediction_preprocessed.nii.gz")

    image_np = data["image"].squeeze(0).cpu().numpy()
    save_overlay(image_np, pred, out_dir / "overlay.png")

    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
