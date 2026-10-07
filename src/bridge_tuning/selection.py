"""Label-free bridge assessment using the definitions in the manuscript."""
import numpy as np
from scipy.spatial.distance import cdist


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
