"""Swin UNETR and bridge full fine-tuning and target LoRA."""
import inspect
from pathlib import Path

import torch
from monai.networks.nets import SwinUNETR
from torch import nn


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
        raise ValueError("Expected a Bridge-Tuning release checkpoint; use convert_checkpoint.py for trusted legacy files")
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
    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint["config"], checkpoint
