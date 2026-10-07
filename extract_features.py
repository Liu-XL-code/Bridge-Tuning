"""Frozen image-only Swin stage-3 features; no segmentation labels are read."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import numpy as np
import torch
from monai import transforms as T
from bridge_tuning.config import load_config, write_json
from bridge_tuning.data import check_geometry, preprocessing, read_manifest
from bridge_tuning.engine import select_device
from bridge_tuning.model import build_model, initialize, read_checkpoint


def extract(checkpoint, manifest, output, device, image_config=None):
    cfg = load_config(image_config) if image_config else read_checkpoint(checkpoint)["config"]
    model = build_model(cfg, apply_peft=False)
    initialize(model, checkpoint)
    model.requires_grad_(False).to(device).eval()
    # Remove even label paths from items before geometry checking or loading images.
    items = [{k: v for k, v in row.items() if k != "label"}
             for row in read_manifest(manifest, labels_required=False)]
    check_geometry(items, cfg["preprocessing"]["geometry_tolerance_mm"])
    tf = T.Compose([preprocessing(cfg, with_label=False),
                    T.Resized(keys="image", spatial_size=tuple(cfg["preprocessing"]["roi"]), mode="area")])
    cache, rows = {}, []
    def hook(module, inputs, output_tensor):
        cache["features"] = output_tensor.detach().mean(dim=(2,3,4)).cpu().numpy()
    handle = model.swinViT.layers3[-1].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            for item in items:
                cache.clear()
                model.swinViT(tf(item)["image"].unsqueeze(0).to(device))
                rows.append(cache["features"][0])
    finally:
        handle.remove()
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix != ".npy":
        raise ValueError("Feature output must have .npy extension")
    np.save(output, np.stack(rows), allow_pickle=False)
    write_json(output.with_suffix(".json"), {"ids": [x["id"] for x in items],
        "shape": list(np.stack(rows).shape), "labels_accessed": False,
        "image_modality_profile": cfg["preprocessing"]["modality"],
        "feature_definition": "Swin stage 3 output, global spatial average",
        "preprocessing": "checkpoint image preprocessing, then whole-volume area resize to ROI"})


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True); p.add_argument("--csv", required=True)
    p.add_argument("--output", required=True); p.add_argument("--device", default="auto")
    p.add_argument("--config", help="Optional CT preprocessing profile for the frozen backbone")
    a=p.parse_args(); torch.set_num_threads(4)
    extract(a.checkpoint, a.csv, a.output, select_device(a.device), a.config)
