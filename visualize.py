"""Show input, ground truth and prediction on the original NIfTI grid."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def render(image_path, label_path, prediction_path, output, synthetic=False):
    image = nib.load(image_path)
    arrays = [nib.load(p) for p in [label_path, prediction_path]]
    if any(a.shape != image.shape or not np.allclose(a.affine, image.affine, atol=1e-3, rtol=0) for a in arrays):
        raise ValueError("All montage inputs must share the same native grid")
    volume = np.asarray(image.dataobj)
    gt, pred = [np.asarray(a.dataobj)>0 for a in arrays]
    if gt.any():
        center = int(np.argmax(gt.sum(axis=(0,1))))
    else:
        center = volume.shape[2]//2
    planes = sorted(set(max(0,min(volume.shape[2]-1,center+d)) for d in [-4,0,4]))
    figure,axes=plt.subplots(len(planes),3,figsize=(9,3*len(planes)),squeeze=False)
    for row,z in enumerate(planes):
        for col in range(3):
            ax=axes[row,col];ax.imshow(volume[:,:,z].T,cmap="gray",vmin=-175,vmax=250,origin="lower")
            if col:
                mask=gt[:,:,z].T if col==1 else pred[:,:,z].T
                overlay=np.ma.masked_where(~mask,mask)
                ax.imshow(overlay,cmap="autumn" if col==1 else "winter",alpha=.5,origin="lower",vmin=0,vmax=1)
            ax.set_axis_off()
            if row==0:ax.set_title(["Synthetic input" if synthetic else "Input","Ground truth","Model prediction"][col])
    if synthetic:
        figure.suptitle("Synthetic software demonstration - not clinical validation",fontsize=11)
    figure.tight_layout()
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(output,dpi=160)
    plt.close(figure)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image",required=True);p.add_argument("--label",required=True)
    p.add_argument("--prediction",required=True);p.add_argument("--output",required=True)
    p.add_argument("--synthetic",action="store_true",help="Label the figure as a synthetic software demonstration")
    a=p.parse_args();render(a.image,a.label,a.prediction,a.output,a.synthetic)
