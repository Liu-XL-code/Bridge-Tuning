"""
单中心30-6-6划分训练器（第一步实验）

功能：
1. 对每个中心的42个病例进行固定的30-6-6划分
2. 从基础模型预训练权重开始微调
3. 使用30个训练样本训练，6个验证样本做早停
4. 在6个测试样本上评估，保存per-case和平均指标
"""

import os
import sys
import json
import csv
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
import argparse

sys.path.append('..')

import lib.models as models
from lib.data.data_datasets import get_dataloader
from lib.utils import SmoothedValue, set_seed
from lib.tools import TrainingLogger, MetricsCalculator, CheckpointManager
from .base_trainer import BaseTrainer
from monai.inferers import sliding_window_inference


class CenterStep1_trainer(BaseTrainer):
    """
    单中心30-6-6划分训练器

    对每个中心的42个病例进行固定划分：
    - 30例：训练集
    - 6例：验证集
    - 6例：测试集（训练过程中不使用）
    """

    def __init__(self, args):
        # 在调用super之前，需要将train下的配置提升到顶层，以便BaseTrainer能正确访问
        # BaseTrainer期望args.epochs和args.batch_size在顶层
        if not hasattr(args, 'epochs'):
            args.epochs = args.train.epochs
        if not hasattr(args, 'batch_size'):
            args.batch_size = args.train.batch_size
        if not hasattr(args, 'lr'):
            args.lr = args.train.lr
        if not hasattr(args, 'optimizer'):
            args.optimizer = args.train.optimizer
        if not hasattr(args, 'weight_decay'):
            args.weight_decay = args.train.weight_decay
        if not hasattr(args, 'warmup_epochs'):
            args.warmup_epochs = getattr(args.train, 'warmup_epochs', 10)
        if not hasattr(args, 'output_dir'):
            args.output_dir = str(args.experiment.output_root)

        super().__init__(args)
        self.args = args

        # 重写学习率（对于小batch_size，不使用batch_size缩放）
        # BaseTrainer的init_lr会缩放：lr * batch_size / 256
        # 对于batch_size=2，这会导致学习率过小，因此直接使用配置的学习率
        self.lr = args.train.lr

        # 获取当前中心配置
        self.current_center = args.current_center
        self.center_configs = args.center_configs
        self.center_config = self.center_configs[self.current_center]

        # 实验配置
        self.center_name = self.current_center
        self.output_root = Path(args.experiment.output_root)
        self.output_dir = self.output_root / f"center_{self.center_name}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 更新output_dir，让BaseTrainer的工具使用正确的目录
        self.args.output_dir = str(self.output_dir)

        # 数据配置（从center_config中获取）
        self.csv_path = self.center_config.csv_path
        self.data_path = self.center_config.data_path
        self.image_key = args.data.image_key  # 通常是'image'
        self.label_key = args.data.label_key  # 通常是'label'

        # 划分配置
        self.split_seed = args.train.split_seed
        self.num_total = args.train.num_total
        self.num_train = args.train.num_train
        self.num_val = args.train.num_val
        self.num_test = args.train.num_test

        # 早停配置
        self.early_stop_patience = args.train.early_stopping_patience
        self.early_stop_threshold = getattr(args.train, 'early_stop_threshold', 0.001)
        self.val_interval = getattr(args.train, 'val_interval', 10)

        # 初始化logger
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = TrainingLogger(log_dir=log_dir, rank=self.rank, distributed=self.distributed)
        self.logger.log_config(args)

        # 初始化metrics计算器
        self.metrics_calc = MetricsCalculator()

        # 更新checkpoint目录
        checkpoint_dir = self.output_dir / 'checkpoints'
        self.ckpt_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            rank=self.rank,
            keep_last_n=3,
            early_stopping_patience=self.early_stop_patience,
            min_delta=self.early_stop_threshold
        )

        # 创建metrics目录
        self.metrics_dir = self.output_dir / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Initialized trainer for center: {self.center_name}")
        self.logger.info(f"Output directory: {self.output_dir}")

    def _split_data(self):
        """
        30-6-6固定划分逻辑

        使用split_seed固定随机顺序，然后按顺序切分：
        - 前30个 → 训练集
        - 接下来6个 → 验证集
        - 最后6个 → 测试集
        """
        self.logger.info("=" * 60)
        self.logger.info("Splitting data into 30-6-6...")

        # 读取CSV文件
        all_samples = []
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_samples.append(row)

        if len(all_samples) != self.num_total:
            raise ValueError(
                f"CSV文件包含{len(all_samples)}个样本，但配置要求{self.num_total}个样本"
            )

        self.logger.info(f"Total samples: {len(all_samples)}")

        # 使用split_seed固定随机顺序
        random.seed(self.split_seed)
        np.random.seed(self.split_seed)
        shuffled_indices = list(range(len(all_samples)))
        random.shuffle(shuffled_indices)

        # 按顺序切分
        train_indices = shuffled_indices[:self.num_train]
        val_indices = shuffled_indices[self.num_train:self.num_train + self.num_val]
        test_indices = shuffled_indices[self.num_train + self.num_val:]

        # 提取样本
        train_samples = [all_samples[i] for i in train_indices]
        val_samples = [all_samples[i] for i in val_indices]
        test_samples = [all_samples[i] for i in test_indices]

        self.logger.info(f"Train: {len(train_samples)} samples")
        self.logger.info(f"Val: {len(val_samples)} samples")
        self.logger.info(f"Test: {len(test_samples)} samples")
        self.logger.info("=" * 60)

        # 从image路径提取patient_id（使用文件名，去掉扩展名）
        def extract_patient_id(sample):
            image_path = sample[self.image_key]
            # 获取文件名（去掉路径和扩展名）
            filename = os.path.basename(image_path)
            patient_id = os.path.splitext(filename)[0]
            return patient_id

        # 保存划分信息
        split_info = {
            'split_seed': self.split_seed,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices,
            'train_patient_ids': [extract_patient_id(s) for s in train_samples],
            'val_patient_ids': [extract_patient_id(s) for s in val_samples],
            'test_patient_ids': [extract_patient_id(s) for s in test_samples],
        }
        split_file = self.output_dir / 'data_split.json'
        with open(split_file, 'w') as f:
            json.dump(split_info, f, indent=2)
        self.logger.info(f"Saved data split info to {split_file}")

        return train_samples, val_samples, test_samples

    def build_model(self):
        """构建模型并加载预训练权重"""
        args = self.args

        self.logger.info("Building model...")

        # 获取模型类
        model_class = getattr(models, args.model.model_name, None)
        if model_class is None:
            raise ValueError(f"Model {args.model.model_name} not found in lib.models")

        # 创建模型（使用center_config中的num_classes）
        self.model = model_class(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.model.in_channels,
            out_channels=self.center_config.num_classes,  # 使用center_config中的num_classes
            feature_size=args.model.feature_size,
        )

        # 加载预训练权重
        self.args.pretrained_path = args.model.pretrained_ckpt
        self._load_pretrained_weights()

        # 包装模型
        self.wrap_model()

        self.logger.info("Model built and pretrained weights loaded")

    def build_dataloader(self):
        """构建数据加载器"""
        args = self.args

        # 划分数据
        train_samples, val_samples, test_samples = self._split_data()

        # 使用配置的data_path
        data_path = self.data_path
        self.logger.info(f"Using data_path: {data_path}")

        # 构建训练集DataLoader
        self.train_loader = get_dataloader(
            data_path=data_path,
            list_file=None,
            sample_list=train_samples,
            batch_size=args.train.batch_size,
            num_workers=self.workers,
            is_train=True,
            distributed=self.distributed,
            args=args,
            use_preload=True
        )

        # 构建验证集DataLoader
        self.val_loader = get_dataloader(
            data_path=data_path,
            list_file=None,
            sample_list=val_samples,
            batch_size=1,  # 验证时batch_size=1
            num_workers=self.workers,
            is_train=False,
            distributed=False,
            args=args,
            use_preload=True
        )

        # 构建测试集DataLoader（保存供测试时使用）
        self.test_loader = get_dataloader(
            data_path=data_path,
            list_file=None,
            sample_list=test_samples,
            batch_size=1,  # 测试时batch_size=1
            num_workers=self.workers,
            is_train=False,
            distributed=False,
            args=args,
            use_preload=True
        )

        self.logger.info(f"Train loader: {len(self.train_loader.dataset)} samples")
        self.logger.info(f"Val loader: {len(self.val_loader.dataset)} samples")
        self.logger.info(f"Test loader: {len(self.test_loader.dataset)} samples")

        # 保存测试样本列表（用于后续per-case结果保存）
        self.test_samples = test_samples

    def _compute_loss(self, outputs, labels):
        """计算分割损失（Dice Loss + CE Loss）"""
        # Dice Loss
        dice_loss = self._dice_loss(outputs, labels)

        # Cross Entropy Loss
        ce_target = labels.squeeze(1) if labels.dim() == 5 and labels.shape[1] == 1 else labels
        ce_loss = F.cross_entropy(outputs, ce_target.long())

        return dice_loss + ce_loss

    def _dice_loss(self, pred, target, smooth=1e-5):
        """Dice Loss"""
        pred = F.softmax(pred, dim=1)

        if target.dim() == 5 and target.shape[1] == 1:
            target = target.squeeze(1)

        target_one_hot = F.one_hot(target.long(), num_classes=pred.shape[1])
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()

        intersection = (pred * target_one_hot).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))

        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()

        loss_meter = SmoothedValue(window_size=self.args.train.print_freq)
        dice_meter = SmoothedValue(window_size=self.args.train.print_freq)
        iou_meter = SmoothedValue(window_size=self.args.train.print_freq)

        for i, batch in enumerate(self.train_loader):
            if self.args.device == 'cuda':
                images = batch['image'].cuda(non_blocking=True)
                labels = batch['label'].cuda(non_blocking=True)
            elif self.args.device == 'cpu':
                images = batch['image']
                labels = batch['label']

            # 前向传播
            outputs = self.wrapped_model(images)
            loss = self._compute_loss(outputs, labels)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 计算指标
            with torch.no_grad():
                dice = self.metrics_calc.dice_score(outputs, labels)
                iou = self.metrics_calc.iou_score(outputs, labels)

            loss_meter.update(loss.item())
            dice_meter.update(dice)
            iou_meter.update(iou)

            if (i + 1) % self.args.train.print_freq == 0:
                self.logger.info(
                    f"Epoch [{epoch}] Batch [{i+1}/{len(self.train_loader)}] | "
                    f"Loss: {loss.item():.4f} | Dice: {dice:.4f} | IoU: {iou:.4f}"
                )

        return {
            'loss': loss_meter.global_avg,
            'dice': dice_meter.global_avg,
            'iou': iou_meter.global_avg,
        }

    def validate(self):
        """验证"""
        self.model.eval()

        dice_scores = []
        iou_scores = []
        hd95_scores = []

        # Sliding window参数
        roi_size = (self.args.roi_x, self.args.roi_y, self.args.roi_z)
        sw_batch_size = getattr(self.args.inference, 'sw_batch_size', 4)
        overlap = getattr(self.args.inference, 'infer_overlap', 0.5)

        with torch.no_grad():
            for batch in self.val_loader:
                if self.args.device == 'cuda':
                    images = batch['image'].cuda(non_blocking=True)
                    labels = batch['label'].cuda(non_blocking=True)
                elif self.args.device == 'cpu':
                    images = batch['image']
                    labels = batch['label']

                # 使用sliding window inference
                outputs = sliding_window_inference(
                    inputs=images,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=self.model,
                    overlap=overlap
                )

                dice_scores.append(self.metrics_calc.dice_score(outputs, labels))
                iou_scores.append(self.metrics_calc.iou_score(outputs, labels))
                hd95_scores.append(self.metrics_calc.hausdorff_distance_95(outputs, labels))

        return {
            'dice': np.mean(dice_scores),
            'iou': np.mean(iou_scores),
            'hd95': np.mean(hd95_scores)
        }

    def test(self):
        """
        在测试集上评估

        返回：
            - 平均指标字典
            - per-case结果列表
        """
        self.logger.info("=" * 60)
        self.logger.info("Testing on test set...")

        # 加载最佳checkpoint
        best_ckpt_path = self.ckpt_manager.checkpoint_dir / 'best_checkpoint.pth'
        if best_ckpt_path.exists():
            self.logger.info(f"Loading best checkpoint from {best_ckpt_path}")
            checkpoint = torch.load(best_ckpt_path, map_location='cpu', weights_only=False)
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            best_epoch = checkpoint.get('epoch', 'unknown')
            best_metric = checkpoint.get('best_metric', 'unknown')
            self.logger.info(f"Best checkpoint: epoch {best_epoch}, metric {best_metric}")
        else:
            self.logger.warning("Best checkpoint not found, using current model")
            best_epoch = self.current_epoch

        self.model.eval()

        # Sliding window参数
        roi_size = (self.args.roi_x, self.args.roi_y, self.args.roi_z)
        sw_batch_size = getattr(self.args.inference, 'sw_batch_size', 4)
        overlap = getattr(self.args.inference, 'infer_overlap', 0.5)

        per_case_results = []
        dice_scores = []
        iou_scores = []
        hd95_scores = []

        with torch.no_grad():
            for idx, batch in enumerate(self.test_loader):
                # 从image路径提取patient_id
                image_path = self.test_samples[idx][self.image_key]
                patient_id = os.path.splitext(os.path.basename(image_path))[0]

                if self.args.device == 'cuda':
                    images = batch['image'].cuda(non_blocking=True)
                    labels = batch['label'].cuda(non_blocking=True)
                elif self.args.device == 'cpu':
                    images = batch['image']
                    labels = batch['label']

                # 使用sliding window inference
                outputs = sliding_window_inference(
                    inputs=images,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=self.model,
                    overlap=overlap
                )

                # 计算指标
                dice = self.metrics_calc.dice_score(outputs, labels)
                iou = self.metrics_calc.iou_score(outputs, labels)
                hd95 = self.metrics_calc.hausdorff_distance_95(outputs, labels)

                dice_scores.append(dice)
                iou_scores.append(iou)
                hd95_scores.append(hd95)

                per_case_results.append({
                    'patient_id': patient_id,
                    'dice': float(dice),
                    'iou': float(iou),
                    'hd95': float(hd95)
                })

                self.logger.info(
                    f"Test case {idx+1}/{len(self.test_loader)} | "
                    f"Patient ID: {patient_id} | "
                    f"Dice: {dice:.4f} | IoU: {iou:.4f} | HD95: {hd95:.4f}"
                )

        # 计算平均指标
        mean_metrics = {
            'mean_dice': float(np.mean(dice_scores)),
            'mean_iou': float(np.mean(iou_scores)),
            'mean_hd95': float(np.mean(hd95_scores)),
        }

        self.logger.info("=" * 60)
        self.logger.info("Test Results Summary:")
        self.logger.info(f"  Mean Dice: {mean_metrics['mean_dice']:.4f}")
        self.logger.info(f"  Mean IoU: {mean_metrics['mean_iou']:.4f}")
        self.logger.info(f"  Mean HD95: {mean_metrics['mean_hd95']:.4f}")
        self.logger.info("=" * 60)

        return mean_metrics, per_case_results, best_epoch

    def _save_test_results(self, mean_metrics, per_case_results, best_epoch):
        """保存测试结果"""
        # 保存per-case结果到CSV
        per_case_file = self.metrics_dir / f'center_{self.center_name}_test_per_case.csv'
        with open(per_case_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['patient_id', 'dice', 'iou', 'hd95'])
            writer.writeheader()
            writer.writerows(per_case_results)
        self.logger.info(f"Saved per-case results to {per_case_file}")

        # 保存汇总结果到JSON
        summary = {
            'center_name': self.center_name,
            'best_epoch': int(best_epoch) if isinstance(best_epoch, (int, np.integer)) else str(best_epoch),
            'mean_dice': mean_metrics['mean_dice'],
            'mean_iou': mean_metrics['mean_iou'],
            'mean_hd95': mean_metrics['mean_hd95'],
            'num_test_samples': len(per_case_results),
        }

        summary_file = self.metrics_dir / f'center_{self.center_name}_test_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        self.logger.info(f"Saved summary to {summary_file}")

    def _save_best_checkpoint_with_center_name(self):
        """保存带中心名称的checkpoint"""
        best_ckpt_path = self.ckpt_manager.checkpoint_dir / 'best_checkpoint.pth'
        if not best_ckpt_path.exists():
            self.logger.warning("Best checkpoint not found, skipping save with center name")
            return

        # 加载checkpoint
        checkpoint = torch.load(best_ckpt_path, map_location='cpu', weights_only=False)

        # 保存为带中心名称的文件
        center_ckpt_name = f"{self.center_name}_bridge_ckpt.pth"
        center_ckpt_path = self.ckpt_manager.checkpoint_dir / center_ckpt_name
        torch.save(checkpoint, center_ckpt_path)
        self.logger.info(f"Saved checkpoint with center name to {center_ckpt_path}")

    def run(self):
        """主训练循环"""
        self.logger.info("=" * 80)
        self.logger.info(f"Starting training for center: {self.center_name}")
        self.logger.info("=" * 80)

        # 设置全局随机种子
        set_seed(self.args.experiment.seed)

        # 构建模型
        self.build_model()

        # 构建优化器
        self.build_optimizer()

        # 构建数据加载器
        self.build_dataloader()

        # 训练循环
        best_val_dice = 0.0
        patience_counter = 0

        for epoch in range(self.epochs):
            self.current_epoch = epoch

            # 调整学习率
            current_lr = self.adjust_learning_rate(epoch)

            # 训练一个epoch
            train_metrics = self.train_epoch(epoch)
            train_metrics['lr'] = current_lr

            # 验证（按val_interval）
            if (epoch + 1) % self.val_interval == 0:
                val_metrics = self.validate()
                current_val_dice = val_metrics['dice']

                # 判断是否是最佳模型
                is_best = current_val_dice > best_val_dice + self.early_stop_threshold
                if is_best:
                    best_val_dice = current_val_dice
                    patience_counter = 0

                    # 保存最佳checkpoint
                    self.save_checkpoint(
                        epoch,
                        is_best=True,
                        metric_value=current_val_dice
                    )

                    self.logger.info(
                        f"Epoch [{epoch+1}] | Best model saved! | "
                        f"Val Dice: {current_val_dice:.4f}"
                    )
                else:
                    patience_counter += 1
                    self.logger.info(
                        f"Epoch [{epoch+1}] | Val Dice: {current_val_dice:.4f} | "
                        f"Best: {best_val_dice:.4f} | Patience: {patience_counter}/{self.early_stop_patience}"
                    )

                # 早停检查
                if patience_counter >= self.early_stop_patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break

                # 记录指标
                train_metrics.update({f'val_{k}': v for k, v in val_metrics.items()})
            else:
                self.logger.info(
                    f"Epoch [{epoch+1}] | Train Loss: {train_metrics['loss']:.4f} | "
                    f"Train Dice: {train_metrics['dice']:.4f}"
                )

            # 记录epoch
            self.logger.log_epoch(epoch, train_metrics)

        # 训练完成，保存带中心名称的checkpoint
        self._save_best_checkpoint_with_center_name()

        # 在测试集上评估
        mean_metrics, per_case_results, best_epoch = self.test()

        # 保存测试结果
        self._save_test_results(mean_metrics, per_case_results, best_epoch)

        self.logger.info("=" * 80)
        self.logger.info("Training completed!")
        self.logger.info("=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Center Step1 Trainer')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    args = parser.parse_args()

    # 加载配置
    from lib.utils import get_conf
    import sys
    sys.argv = ['', args.config]
    config = get_conf()

    # 创建trainer并运行
    trainer = CenterStep1_trainer(config)
    trainer.run()


if __name__ == '__main__':
    main()
