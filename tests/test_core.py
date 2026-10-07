import copy
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import nibabel as nib
import numpy as np
import torch

from bridge_tuning.config import DEFAULT, validate
from bridge_tuning.data import assert_disjoint, binarize, check_geometry, preprocessing, read_manifest
from bridge_tuning.engine import learning_rate, should_stop
from bridge_tuning.metrics import binary_metrics
from bridge_tuning.model import LoRALinear, apply_strategy, build_model, merged_state
from bridge_tuning.splits import make_folds


class ReleaseTests(unittest.TestCase):
    def test_lora_merge_preserves_outputs(self):
        original = torch.nn.Linear(8, 12)
        layer = LoRALinear(original, 3, 6, 0.1)
        torch.nn.init.normal_(layer.lora_b.weight, std=0.2)
        layer.eval()
        net = torch.nn.Sequential(layer)
        x = torch.randn(2, 4, 8)
        state = merged_state(net)
        merged = torch.nn.Sequential(torch.nn.Linear(8, 12))
        merged.load_state_dict(state, strict=True)
        torch.testing.assert_close(net(x), merged(x), rtol=1e-5, atol=1e-6)

    def test_decoder_and_encoder_training_scope(self):
        cfg = copy.deepcopy(DEFAULT)
        cfg["model"]["feature_size"] = 12
        for mode in ["fft", "lora"]:
            cfg["model"]["strategy"] = mode
            model = build_model(cfg)
            params = dict(model.named_parameters())
            self.assertTrue(params["out.conv.conv.weight"].requires_grad)
            self.assertEqual(params["swinViT.patch_embed.proj.weight"].requires_grad, mode == "fft")
            self.assertEqual(params["encoder1.layer.conv1.conv.weight"].requires_grad, mode == "fft")
            if mode == "lora":
                self.assertTrue(any("lora_a" in k and p.requires_grad for k,p in params.items()))
            del model

    def test_foreground_hd95_units_and_zero_dice_kept(self):
        gt = np.zeros((24,24,24),dtype=np.uint8);gt[6:14,6:14,6:14]=1
        shifted = np.roll(gt, 1, axis=0)
        a = binary_metrics(shifted, gt)
        b = binary_metrics(shifted, gt, spacing=(2,1,1))
        self.assertAlmostEqual(a["hd95"], 1)
        self.assertAlmostEqual(b["hd95"], 2)
        empty = binary_metrics(np.zeros_like(gt), gt)
        self.assertLess(empty["dice"], 1e-6)
        self.assertEqual(empty["hd95"], 100)
        self.assertEqual(binary_metrics(gt, gt)["dice"], 1)

    def test_fixed_subject_folds_and_seeded_supports(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            fixture = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.uint8), np.eye(4))
            lines = ["id,subject_id,image,label"]
            for index in range(12):
                image = p / f"image_{index}.nii.gz"
                label = p / f"label_{index}.nii.gz"
                nib.save(fixture, str(image)); nib.save(fixture, str(label))
                lines.append(f"case{index},subject{index},{image},{label}")
            example = p / "all.csv"
            example.write_text("\n".join(lines) + "\n")
            records = make_folds(example, 3, 3, [0,1,2], 1024, tmp)
            all_tests = []
            for fold in range(3):
                rows = [r for r in records if r["fold"] == fold]
                self.assertTrue(all(r["test_ids"] == rows[0]["test_ids"] for r in rows))
                for r in rows:
                    self.assertFalse(set(r["train_subjects"]) & set(r["test_subjects"]))
                    self.assertEqual(len(r["train_ids"]), 3)
                all_tests += rows[0]["test_ids"]
            self.assertEqual(len(all_tests), len(set(all_tests)))
            self.assertEqual(len(all_tests), 12)
            self.assertTrue(any(records[i]["train_ids"] != records[i+1]["train_ids"] for i in [0,3,6]))

    def test_preprocessing_geometry_and_label_union(self):
        cfg = copy.deepcopy(DEFAULT)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            image = np.zeros((7,8,9),dtype=np.float32)
            image[0,0,0] = -1000;image[1,1,1] = 400;image[3,3,3] = -175
            label = np.zeros_like(image,dtype=np.uint8);label[2,2,2] = 2
            affine = np.diag([1.5,1.5,2,1])
            for a,n in [(image,"image"),(label,"label")]:
                obj = nib.Nifti1Image(a,affine);obj.header.set_xyzt_units("mm")
                nib.save(obj,str(p/(n+".nii.gz")))
            item = {"id":"test","image":str(p/'image.nii.gz'),"label":str(p/'label.nii.gz')}
            check_geometry([item])
            data = preprocessing(cfg)(item)
            self.assertEqual(data["image"].shape, data["label"].shape)
            self.assertAlmostEqual(float(data["image"].max()), 1)
            self.assertAlmostEqual(float(data["image"].min()), 0)
            self.assertEqual(set(torch.unique(data["label"]).tolist()), {0.0,1.0})
            torch.testing.assert_close(data["image"].affine[:3,:3].diagonal(), torch.tensor([1.5,1.5,2],dtype=torch.float64))
            shifted = affine.copy();shifted[2,3] = 2
            nib.save(nib.Nifti1Image(label,shifted),str(p/'label.nii.gz'))
            with self.assertRaisesRegex(ValueError,"grids differ"):
                check_geometry([item])

    def test_split_overlap_and_plateau_guard(self):
        x = {"image":"a.nii.gz","subject_id":"same"}
        with self.assertRaises(ValueError):
            assert_disjoint([x],[{"image":"b.nii.gz","subject_id":"same"}])
        self.assertFalse(should_stop([1.0]*39,DEFAULT["training"]))
        self.assertTrue(should_stop([1.0]*40,DEFAULT["training"]))
        with self.assertRaises(ValueError):
            invalid = copy.deepcopy(DEFAULT);invalid["preprocessing"]["roi"] = [32,32,32];validate(invalid)

    def test_label_rules_learning_rate_and_image_only_access(self):
        x=torch.tensor([0.,.2,.5,1.,2.,3.])
        self.assertEqual(binarize(x,"probability",threshold=.5).tolist(),[0,0,1,1,1,1])
        self.assertEqual(binarize(x,"values",values=[1]).tolist(),[0,0,0,1,0,0])
        t=copy.deepcopy(DEFAULT["training"]);t["schedule"]="constant";t["warmup_epochs"]=0
        self.assertEqual(learning_rate(50,t),t["learning_rate"])
        with tempfile.TemporaryDirectory()as tmp:
            example = Path(tmp) / "image.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros((4,4,4), dtype=np.float32), np.eye(4)), str(example))
            csv=Path(tmp)/"unlabeled.csv"
            csv.write_text(f"id,image,label\ncase,{example},label_does_not_exist.nii.gz\n")
            rows=read_manifest(csv,labels_required=False)
            self.assertNotIn("label",rows[0])


if __name__=="__main__":
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
