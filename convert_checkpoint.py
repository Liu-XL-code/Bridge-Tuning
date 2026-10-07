"""Explicit conversion of locally trusted historical files; excludes optimizers and paths."""
import argparse
import hashlib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import torch
from bridge_tuning.config import load_config


def convert(source, config, destination, role, trusted, backbone_only=False):
    if not trusted:
        raise ValueError("Legacy conversion requires --trusted because legacy torch.load may use pickle")
    ck = torch.load(source, map_location="cpu", weights_only=False)
    raw = ck.get("state_dict", ck.get("model", ck)) if isinstance(ck, dict) else ck
    if not isinstance(raw, dict):
        raise ValueError("Legacy file does not contain a state dictionary")
    state = {}
    for key, value in raw.items():
        if not isinstance(value, torch.Tensor):
            continue
        while key.startswith("module.") or key.startswith("model."):
            key = key.split(".", 1)[1]
        if backbone_only and (key.startswith("out.") or key.startswith("classifier.")):
            continue
        state[key] = value.detach().cpu().contiguous()
    if not state:
        raise ValueError("No model tensors found")
    digest = hashlib.sha256()
    with Path(source).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(b)
    payload = {"schema_version": 1, "state_dict": state, "config": load_config(config),
               "role": role, "epoch": int(ck.get("epoch", 0)) if isinstance(ck, dict) else 0,
               "metadata": {"source_file_sha256": digest.hexdigest(), "optimizer_removed": True,
                            "source_paths_removed": True}}
    payload["metadata"]["initialization_only"] = backbone_only
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    print(f"Converted {len(state)} tensors; optimizer and source paths excluded")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--role", required=True)
    p.add_argument("--trusted", action="store_true")
    p.add_argument("--backbone-only", action="store_true", help="Drop upstream classifier/head; initialization only")
    a = p.parse_args()
    convert(a.input, a.config, a.output, a.role, a.trusted, a.backbone_only)
