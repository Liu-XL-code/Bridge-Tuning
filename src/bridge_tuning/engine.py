"""Explicit training, validation and held-out evaluation. Test labels never enter training."""
import copy
import math
import random
import shutil
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai import transforms as T
from monai.data import DataLoader, Dataset, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.utils import set_determinism

from .config import write_json
from .data import assert_disjoint, augmentation, check_geometry, preprocessing
from .metrics import binary_metrics, summarize
from .model import apply_strategy, build_model, initialize, save_checkpoint


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
