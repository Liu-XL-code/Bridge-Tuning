"""Foreground Dice and symmetric pooled-surface HD95. No low-Dice exclusion."""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


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
