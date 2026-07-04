#!/usr/bin/env python
"""Unified 3D Dice/HD95 evaluation, matching the Bridge-Tuning harness exactly.
Eval space = RAS + Spacing(1.5,1.5,2.0) + intensity window + CropForeground(by image).
Dice/HD95 via the project's MetricsCalculator (HD95 default spacing=voxel units, as in Table II;
an extra hd95_mm uses physical spacing). Predictions are expected in ORIGINAL image space.
Appends per-case rows and a per-(target,seed) summary line.
"""
import os, json, argparse, importlib.util, csv
import numpy as np
import torch
from monai import transforms as T

def load_metrics(code_dir):
    spec = importlib.util.spec_from_file_location("btmetrics", os.path.join(code_dir,"lib","tools","metrics.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.MetricsCalculator

def build_tf(sx=1.5, sy=1.5, sz=2.0, a_min=-175.0, a_max=250.0):
    return T.Compose([
        T.LoadImaged(keys=["image","gt","pred"]),
        T.EnsureChannelFirstd(keys=["image","gt","pred"]),
        T.Orientationd(keys=["image","gt","pred"], axcodes="RAS"),
        T.Spacingd(keys=["image","gt","pred"], pixdim=(sx,sy,sz), mode=("bilinear","nearest","nearest")),
        T.ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
        T.CropForegroundd(keys=["image","gt","pred"], source_key="image"),
        T.ToTensord(keys=["image","gt","pred"]),
    ])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--dsid", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--code_dir", required=True)
    ap.add_argument("--out_percase", required=True)
    ap.add_argument("--out_summary", required=True)
    args = ap.parse_args()

    MetricsCalculator = load_metrics(args.code_dir)
    idx = json.load(open(args.index))[str(args.dsid)]
    target, seed, shot = idx["target"], idx["seed"], idx["shot"]
    tf = build_tf()

    rows = []
    for c in idx["test_cases"]:
        cid = c["case_id"]
        pred_path = os.path.join(args.pred_dir, f"{cid}.nii.gz")
        if not os.path.exists(pred_path):
            rows.append([args.method,target,shot,seed,cid,"","","",  "", "", "missing_pred"]); continue
        try:
            d = tf({"image":c["image"], "gt":c["gt"], "pred":pred_path})
            gt = (d["gt"]>0).float()      # (1,H,W,D): leading 1 acts as batch -> (B=1,H,W,D) label map
            pr = (d["pred"]>0).float()
            # pass dim-4 label maps (B,H,W,D). Do NOT add a channel dim: dice_score/hd95 would then
            # treat it as (B,C,H,W,D) logits and argmax over the singleton channel -> all zeros.
            dice   = MetricsCalculator.dice_score(pr.long(), gt.long(), num_classes=2)
            hd95   = MetricsCalculator.hausdorff_distance_95(pr.long(), gt.long(), num_classes=2)
            hd95mm = MetricsCalculator.hausdorff_distance_95(pr.long(), gt.long(), num_classes=2, voxel_spacing=(1.5,1.5,2.0))
            ge = int(gt.sum().item()==0); pe = int(pr.sum().item()==0)
            rows.append([args.method,target,shot,seed,cid,f"{dice:.6f}",f"{hd95:.4f}",f"{hd95mm:.4f}",ge,pe,"ok"])
        except Exception as e:
            rows.append([args.method,target,shot,seed,cid,"","","","","",f"err:{type(e).__name__}:{str(e)[:60]}"])

    # append per-case
    new = not os.path.exists(args.out_percase)
    with open(args.out_percase,"a",newline="") as f:
        w=csv.writer(f)
        if new: w.writerow(["method","target","shot","seed","case_id","dice","hd95","hd95_mm","gt_empty","pred_empty","status"])
        w.writerows(rows)

    # summary over ok cases
    ok=[r for r in rows if r[10]=="ok"]
    dices=[float(r[5]) for r in ok]; hd=[float(r[6]) for r in ok]; hdmm=[float(r[7]) for r in ok]
    n=len(ok)
    def ms(x): return (float(np.mean(x)), float(np.std(x))) if x else (float("nan"),float("nan"))
    dm,ds=ms(dices); hm,hs=ms(hd); hmm,hmms=ms(hdmm)
    news = not os.path.exists(args.out_summary)
    with open(args.out_summary,"a",newline="") as f:
        w=csv.writer(f)
        if news: w.writerow(["method","target","shot","seed","dice_mean","dice_std","hd95_mean","hd95_std","hd95mm_mean","hd95mm_std","n_cases","n_missing","status"])
        w.writerow([args.method,target,shot,seed,f"{dm:.4f}",f"{ds:.4f}",f"{hm:.3f}",f"{hs:.3f}",f"{hmm:.3f}",f"{hmms:.3f}",n,len(rows)-n,"ok" if n>0 else "no_ok_cases"])
    print(f"[eval] {args.method} {target} {shot}-shot seed{seed}: Dice {dm:.4f}±{ds:.4f}  HD95 {hm:.2f}±{hs:.2f} (vox)  n={n} missing={len(rows)-n}")

if __name__ == "__main__":
    main()
