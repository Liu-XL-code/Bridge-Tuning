"""Cross-center testing trainer.

Trains on a single user-selected center (using predefined train/val/test splits)
and evaluates the resulting checkpoint on every center's test set via
sliding-window inference.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from omegaconf import OmegaConf

import lib.models as models
from lib.data.data_datasets import get_dataloader
from lib.tools import TrainingLogger, CheckpointManager
from lib.utils import SmoothedValue
from .base_trainer import BaseTrainer


class CrossCenterTestTrainer(BaseTrainer):
	"""Train on one center and test on all centers."""

	def __init__(self, args):
		super().__init__(args)
		self.args = args

		self.center_configs = self._to_plain_dict(
			getattr(args, "enter_configs", getattr(args, "center_configs", None))
		)
		if not self.center_configs:
			raise ValueError("Center configurations (enter_configs) are required for CrossCenterTestTrainer")

		self.train_center = getattr(args, "current_center", getattr(args, "train_center", None))
		if not self.train_center:
			# Fallback to the first configured center to avoid hard failures
			self.train_center = next(iter(self.center_configs))
			print(
				f"Warning: current_center not specified. Falling back to '{self.train_center}'. "
				"Set --current_center in the config/CLI to choose another center."
			)
		if self.train_center not in self.center_configs:
			raise ValueError(f"Unknown center '{self.train_center}'. Available centers: {list(self.center_configs.keys())}")
		self.center_config = self.center_configs[self.train_center]

		# Resolve pretrained path (support named shortcuts defined in config)
		raw_weight_bank = getattr(args, "weight_paths", getattr(args, "peight_paths", None))
		weight_bank = self._to_plain_dict(raw_weight_bank) if raw_weight_bank else {}
		pretrained_key = getattr(args, "pretrained_path", None)
		resolved_pretrained = weight_bank.get(pretrained_key, pretrained_key)
		if not resolved_pretrained:
			raise ValueError("Unable to resolve pretrained weights path; please set pretrained_path correctly")
		self.pretrained_path = resolved_pretrained
		self.args.pretrained_path = resolved_pretrained

		# Override LR scaling (use config value directly)
		self.lr = args.lr

		self.val_interval = getattr(args, "val_interval", 10)
		self.print_freq = getattr(args, "print_freq", getattr(args, "finetune_print_freq", 20))
		self.early_stop_enabled = getattr(args, "early_stop_enabled", True)
		self.early_stop_patience = getattr(args, "early_stop_patience", 10)
		self.early_stop_threshold = getattr(args, "early_stop_threshold", 0.01)

		self.run_timestamp = None
		self.run_dir = None
		self.logger = None
		self.best_checkpoint_path = None
		self.best_epoch = -1
		self.best_val_dice = -float("inf")

		# Will be reset once run directory is known
		self.ckpt_manager = None

	@staticmethod
	def _to_plain_dict(cfg):
		"""Convert OmegaConf containers to plain Python dicts when needed."""
		if cfg is None:
			return None
		if isinstance(cfg, dict):
			return cfg
		try:
			return OmegaConf.to_container(cfg, resolve=True)
		except Exception:
			return cfg

	def build_model(self):
		"""Create the SwinUNETR model and load pretrained weights."""
		model_class = getattr(models, self.args.model_name, None)
		if model_class is None:
			raise ValueError(f"Model {self.args.model_name} not found in lib.models")

		self.model = model_class(
			img_size=(self.args.roi_x, self.args.roi_y, self.args.roi_z),
			in_channels=self.args.in_channels,
			out_channels=self.center_config["num_classes"],
			feature_size=self.args.feature_size,
		)

		self._load_pretrained_weights(self.pretrained_path)
		self.wrap_model()

	def _load_pretrained_weights(self, weight_path=None):
		"""Load pretrained weights while skipping output layers."""
		load_path = weight_path or self.pretrained_path
		if not load_path or not os.path.exists(load_path):
			msg = f"Pretrained model not found at {load_path}"
			if self.logger:
				self.logger.warning(msg)
			else:
				print(f"Warning: {msg}")
			return

		checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)
		state_dict = checkpoint.get("state_dict", checkpoint)

		filtered_state = {}
		for key, value in state_dict.items():
			clean_key = key[7:] if key.startswith("module.") else key
			if not clean_key.startswith("model."):
				clean_key = f"model.{clean_key}"
			if "out.conv" in clean_key or "classifier" in clean_key:
				continue
			filtered_state[clean_key] = value

		incompatible = self.model.load_state_dict(filtered_state, strict=False)
		missing = [k for k in incompatible.missing_keys if "out.conv" not in k and "classifier" not in k]
		unexpected = [k for k in incompatible.unexpected_keys if "out.conv" not in k and "classifier" not in k]
		if self.logger:
			self.logger.info(f"Loaded pretrained weights from {load_path}")
			if missing:
				self.logger.warning(f"Missing keys (excluding head): {len(missing)}")
			if unexpected:
				self.logger.warning(f"Unexpected keys (excluding head): {len(unexpected)}")

	def run(self):
		"""Main training + cross-center testing pipeline."""
		self._prepare_run_environment()
		self._build_dataloaders()

		print("\n" + "=" * 80)
		print("Cross-Center Training")
		print("=" * 80)
		print(f"Training center      : {self.train_center}")
		print(f"Validation interval  : {self.val_interval} epochs")
		print(f"Early stop patience  : {self.early_stop_patience} validations")
		print(f"Pretrained checkpoint: {self.pretrained_path}")
		print("=" * 80 + "\n")

		best_ckpt = self._train_with_validation()
		self._load_checkpoint_for_eval(best_ckpt)
		center_results = self._evaluate_all_centers()
		self._save_test_summary(center_results)

		print("\n" + "=" * 80)
		print("Cross-center evaluation completed")
		print(f"Results saved to: {self.run_dir}")
		print("=" * 80 + "\n")

	def _prepare_run_environment(self):
		"""Create run-specific directories, logger, and checkpoint manager."""
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		self.run_timestamp = f"{timestamp}_{self.train_center}"
		self.run_dir = Path(self.args.output_dir) / self.run_timestamp
		log_dir = self.run_dir / "logs"
		log_dir.mkdir(parents=True, exist_ok=True)

		self.logger = TrainingLogger(log_dir=log_dir, rank=self.rank, distributed=self.distributed)
		self.logger.log_config(self.args)

		self.ckpt_manager = CheckpointManager(
			checkpoint_dir=self.run_dir / "checkpoints",
			rank=self.rank,
			keep_last_n=1,
			early_stopping_patience=None,
		)

	def _build_dataloaders(self):
		"""Build train/validation loaders for the selected center."""
		cfg = self.center_config
		self.train_loader = get_dataloader(
			data_path=cfg["data_path"],
			list_file=cfg["train_csv"],
			sample_list=None,
			batch_size=self.args.batch_size,
			num_workers=self.workers,
			is_train=True,
			distributed=self.distributed,
			args=self.args,
			use_preload=True,
		)
		self.val_loader = get_dataloader(
			data_path=cfg["data_path"],
			list_file=cfg["validation_csv"],
			sample_list=None,
			batch_size=1,
			num_workers=self.workers,
			is_train=False,
			distributed=False,
			args=self.args,
			use_preload=True,
		)

	def _train_with_validation(self):
		"""Train with periodic validation and return best checkpoint path."""
		best_path = None
		patience_counter = 0

		for epoch in range(self.args.epochs):
			self.current_epoch = epoch
			current_lr = self.adjust_learning_rate(epoch)
			train_metrics = self.train_epoch(epoch)
			epoch_metrics = {**train_metrics, "lr": current_lr}

			should_validate = ((epoch + 1) % self.val_interval == 0) or (epoch == 0) or (epoch == self.args.epochs - 1)
			if should_validate:
				val_metrics = self.validate()
				epoch_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
				improvement = val_metrics["dice"] - self.best_val_dice
				if improvement > self.early_stop_threshold or self.best_epoch < 0:
					self.best_val_dice = val_metrics["dice"]
					self.best_epoch = epoch
					patience_counter = 0
					state = {
						"epoch": epoch + 1,
						"arch": self.args.arch,
						"state_dict": self.model.state_dict(),
						"optimizer": self.optimizer.state_dict(),
						"val_dice": self.best_val_dice,
					}
					self.ckpt_manager.save_checkpoint(state, epoch, is_best=True, metric_value=self.best_val_dice)
					best_path = self.ckpt_manager.checkpoint_dir / "best_checkpoint.pth"
					if self.logger:
						self.logger.info(f"New best validation Dice {self.best_val_dice:.4f} at epoch {epoch}")
				else:
					patience_counter += 1
					if self.early_stop_enabled and patience_counter >= self.early_stop_patience:
						if self.logger:
							self.logger.info(
								f"Early stopping triggered after {patience_counter} validations without sufficient improvement"
							)
						break

			if self.logger:
				self.logger.log_epoch(epoch, epoch_metrics)

		return best_path

	def train_epoch(self, epoch):
		"""Train for one epoch on the selected center."""
		self.model.train()
		loss_meter = SmoothedValue(window_size=self.print_freq)
		dice_meter = SmoothedValue(window_size=self.print_freq)
		iou_meter = SmoothedValue(window_size=self.print_freq)

		for step, batch in enumerate(self.train_loader):
			if self.args.device == "cuda":
				images = batch["image"].cuda(non_blocking=True)
				labels = batch["label"].cuda(non_blocking=True)
			else:
				images = batch["image"]
				labels = batch["label"]

			outputs = self.wrapped_model(images)
			loss = self._compute_loss(outputs, labels)

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			with torch.no_grad():
				dice = self.metrics_calc.dice_score(outputs, labels)
				iou = self.metrics_calc.iou_score(outputs, labels)

			loss_meter.update(loss.item())
			dice_meter.update(dice)
			iou_meter.update(iou)

			if self.logger and (step + 1) % self.print_freq == 0:
				self.logger.info(
					f"Epoch {epoch:03d} | Iter {step + 1:04d}/{len(self.train_loader)} | "
					f"Loss: {loss.item():.4f} | Dice: {dice:.4f} | IoU: {iou:.4f}"
				)

		return {
			"loss": loss_meter.global_avg,
			"dice": dice_meter.global_avg,
			"iou": iou_meter.global_avg,
		}

	def _compute_loss(self, outputs, labels):
		"""Dice + CE segmentation loss."""
		dice_loss = self._dice_loss(outputs, labels)
		ce_target = labels.squeeze(1) if labels.dim() == 5 and labels.shape[1] == 1 else labels
		ce_loss = F.cross_entropy(outputs, ce_target.long())
		return dice_loss + ce_loss

	def _dice_loss(self, pred, target, smooth=1e-5):
		pred = F.softmax(pred, dim=1)
		if target.dim() == 5 and target.shape[1] == 1:
			target = target.squeeze(1)
		target_one_hot = F.one_hot(target.long(), num_classes=pred.shape[1])
		target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()
		intersection = (pred * target_one_hot).sum(dim=(2, 3, 4))
		union = pred.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))
		dice = (2.0 * intersection + smooth) / (union + smooth)
		return 1.0 - dice.mean()

	def validate(self):
		"""Run sliding-window validation."""
		return self._evaluate_loader(self.val_loader, split_name="val")

	def _load_checkpoint_for_eval(self, checkpoint_path):
		if not checkpoint_path or not os.path.exists(checkpoint_path):
			msg = "Best checkpoint not found; evaluation will use final model weights"
			if self.logger:
				self.logger.warning(msg)
			else:
				print(f"Warning: {msg}")
			return

		checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
		state_dict = checkpoint.get("state_dict", checkpoint)
		target_model = self.wrapped_model.module if hasattr(self.wrapped_model, "module") else self.wrapped_model
		target_model.load_state_dict(state_dict)
		if self.logger:
			self.logger.info(f"Loaded best checkpoint from {checkpoint_path}")

	def _evaluate_all_centers(self):
		"""Evaluate the trained model on every center's test split."""
		results = {}
		for center_name, cfg in self.center_configs.items():
			test_loader = get_dataloader(
				data_path=cfg["data_path"],
				list_file=cfg["test_csv"],
				sample_list=None,
				batch_size=1,
				num_workers=self.workers,
				is_train=False,
				distributed=False,
				args=self.args,
				use_preload=True,
			)
			metrics = self._evaluate_loader(test_loader, split_name=f"test_{center_name}")
			results[center_name] = metrics
		return results

	def _evaluate_loader(self, loader, split_name):
		"""Shared sliding-window inference for validation/test splits."""
		self.model.eval()
		dice_scores, iou_scores, hd95_scores = [], [], []
		skipped = 0
		total_samples = len(loader.dataset) if hasattr(loader, "dataset") else 0

		roi_size = (self.args.roi_x, self.args.roi_y, self.args.roi_z)
		sw_batch_size = getattr(self.args, "sw_batch_size", 4)
		overlap = getattr(self.args, "infer_overlap", 0.5)

		with torch.no_grad():
			for batch in loader:
				if self.args.device == "cuda":
					images = batch["image"].cuda(non_blocking=True)
					labels = batch["label"].cuda(non_blocking=True)
				else:
					images = batch["image"]
					labels = batch["label"]

				outputs = sliding_window_inference(
					inputs=images,
					roi_size=roi_size,
					sw_batch_size=sw_batch_size,
					predictor=self.model,
					overlap=overlap,
				)

				dice = self.metrics_calc.dice_score(outputs, labels)
				if dice <= 1e-6 or np.isnan(dice) or np.isinf(dice):
					skipped += 1
					if self.logger:
						self.logger.warning(
							f"[{split_name}] Skipped sample with invalid dice={dice:.6f} (total skipped: {skipped})"
						)
					continue

				dice_scores.append(dice)
				iou_scores.append(self.metrics_calc.iou_score(outputs, labels))
				hd95_scores.append(self.metrics_calc.hausdorff_distance_95(outputs, labels))

		if not dice_scores:
			return {
				"dice": 0.0,
				"iou": 0.0,
				"hd95": float("inf"),
				"samples_skipped": skipped,
				"total_samples": total_samples,
			}

		return {
			"dice": float(np.mean(dice_scores)),
			"iou": float(np.mean(iou_scores)),
			"hd95": float(np.mean(hd95_scores)),
			"samples_skipped": skipped,
			"total_samples": total_samples,
		}

	def _save_test_summary(self, center_results):
		"""Persist aggregated cross-center metrics."""
		summary = {
			"train_center": self.train_center,
			"best_epoch": self.best_epoch,
			"best_val_dice": float(self.best_val_dice) if self.best_epoch >= 0 else None,
			"results": center_results,
		}
		output_path = self.run_dir / "cross_center_results.json"
		with open(output_path, "w") as f:
			json.dump(summary, f, indent=2)
		if self.logger:
			self.logger.info(f"Saved cross-center summary to {output_path}")
