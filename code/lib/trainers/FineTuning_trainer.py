import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import math
import csv
import tempfile
import sys
sys.path.append('..')

import lib.models as models
from lib.data.data_datasets import get_dataloader
from lib.utils import SmoothedValue
from lib.tools import TrainingLogger
from .base_trainer import BaseTrainer
from monai.inferers import sliding_window_inference
class FineTuning_trainer(BaseTrainer):
    """
    微调训练器
    直接在私有数据集上依次进行Fine-tuning
    """
    def __init__(self, args):
        super().__init__(args)
        self.args=args
        self.output_dir = args.output_dir
        k_shot_config = getattr(args, 'k_shot', 5)
        # 数据集配置
        self.finetune_datasets = args.finetune_datasets

        if isinstance(k_shot_config, (list, tuple)) or hasattr(k_shot_config, '__iter__') and not isinstance(k_shot_config, str):
            # 如果是列表类型（包括ListConfig），转换为Python list
            self.k_shot_list = list(k_shot_config)
        else:
            # 如果是单个值，包装成列表
            self.k_shot_list = [k_shot_config]

        #随机种子配置：用于数据采样的不同种子（产生不同的样本组合）
        random_seeds_config = getattr(args, 'random_seeds', [0, 1, 2])
        self.random_seeds = list(random_seeds_config) if hasattr(random_seeds_config, '__iter__') else [random_seeds_config]

        # 全局训练种子：用于确保训练过程的可重复性
        self.global_train_seed = getattr(args, 'seed', 42)

        # 早停配置
        self.early_stop_patience = getattr(args, 'early_stop_patience', 10)
        self.early_stop_threshold = getattr(args, 'early_stop_threshold', 0.01)
        self.early_stop_enabled = getattr(args, 'early_stop_enabled', True)

    def build_model(self):
        """构建模型"""
        args = self.args

        print(f"Creating model architecture")

        model_class = getattr(models, args.model_name, None)
        if model_class is None:
            raise ValueError(f"Model {args.model_name} not found in lib.models")

        # 创建模型
        self.model = model_class(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=self.finetune_datasets[0]['num_classes'],
            feature_size=args.feature_size,
        )

        # 初始加载预训练权重
        self._load_pretrained_weights()

        self.wrap_model()

    def build_dataloader(self,dataset_config, k_shot, seed):
        args = self.args

        # K-shot采样
        finetune_csv = dataset_config['finetune_csv']
        with open(finetune_csv, 'r') as f:
            all_samples = list(csv.DictReader(f))

        # 使用采样种子选择样本
        random.seed(seed)
        np.random.seed(seed)
        selected_samples = random.sample(all_samples, min(k_shot, len(all_samples)))

        self.logger.info(f"Selected {len(selected_samples)} samples from {len(all_samples)} for training")

        # ✅ 直接传入样本列表，不需要创建临时CSV！
        self.train_loader = get_dataloader(
            data_path=dataset_config['data_path'],
            list_file=None,
            sample_list=selected_samples,  # 直接传列表
            batch_size=args.batch_size,
            num_workers=self.workers,
            is_train=True,
            distributed=self.distributed,
            args=args,
            use_preload=True
        )

        # 测试集仍然使用CSV文件（完整数据）
        self.test_loader = get_dataloader(
            data_path=dataset_config['data_path'],
            list_file=dataset_config['test_csv'],  # 使用原始CSV
            sample_list=None,
            batch_size=1,
            num_workers=self.workers,
            is_train=False,
            distributed=False,
            args=args,
            use_preload=True
        )

        self.logger.info(f"Train samples: {len(self.train_loader.dataset)}, Test samples: {len(self.test_loader.dataset)}")


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

        loss_meter = SmoothedValue(window_size=self.args.finetune_print_freq)
        dice_meter = SmoothedValue(window_size=self.args.finetune_print_freq)
        iou_meter = SmoothedValue(window_size=self.args.finetune_print_freq)
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

            # 计算所有指标
            with torch.no_grad():
                dice = self.metrics_calc.dice_score(outputs, labels)
                iou = self.metrics_calc.iou_score(outputs, labels)

            loss_meter.update(loss.item())
            dice_meter.update(dice)
            iou_meter.update(iou)

            self.logger.info(
                f"Epoch: {epoch:03d} | Loss: {loss.item():.4f} | Dice: {dice:.4f} | IoU: {iou:.4f}"
            )

        return {
            'loss': loss_meter.global_avg,
            'dice': dice_meter.global_avg,
            'iou': iou_meter.global_avg,
        }

    def test(self):
        self.model.eval()
        dice_scores, iou_scores, hd95_scores = [], [], []
        # Sliding window参数
        roi_size = (self.args.roi_x, self.args.roi_y, self.args.roi_z)
        sw_batch_size = getattr(self.args, 'sw_batch_size', 4)
        overlap = getattr(self.args, 'infer_overlap', 0.5)

        with torch.no_grad():
            for batch in self.test_loader:
                if self.args.device == 'cuda':
                    images = batch['image'].cuda(non_blocking=True)
                    labels = batch['label'].cuda(non_blocking=True)
                elif self.args.device == 'cpu':
                    images = batch['image']
                    labels = batch['label']

                # 使用sliding window inference覆盖整张图像
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



    def run(self):
        print("\n" + "=" * 80)
        print("Fine-tuning on Private Datasets")
        print("=" * 80)
        print(f"K-shot values: {self.k_shot_list}")
        print(f"Random seeds: {self.random_seeds}")
        print(f"Global training seed: {self.global_train_seed}")
        print("=" * 80)
         # 为整个运行创建一个时间戳（只创建一次）
        from datetime import datetime
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        print(f"Run timestamp: {self.run_timestamp}")
        print("=" * 80)
        # 对每个数据集
        for dataset_config in self.finetune_datasets:
            self._finetune_on_dataset(dataset_config)

    def _finetune_on_dataset(self, dataset_config):
        """在单个数据集上进行微调，支持多K值和多随机种子"""
        dataset_name = dataset_config['name']
        print("\n" + "=" * 80)
        print(f"Dataset: {dataset_name}")
        print("=" * 80)
        # 收集当前k-shot所有种子的结果
         # 只构建一次test_loader来测试预训练模型
        self.test_loader = get_dataloader(
            data_path=dataset_config['data_path'],
            list_file=dataset_config['test_csv'],
            sample_list=None,
            batch_size=1,
            num_workers=self.workers,
            is_train=False,
            distributed=False,
            args=self.args,
            use_preload=True
        )
        pretrained_results = self.test()

        # 对每个k_shot值分别处理
        for k_shot in self.k_shot_list:
            print(f"\n>>> K-shot: {k_shot}")
            k_shot_results = []  # 为每个k_shot单独收集结果

            for seed in self.random_seeds:
                print(f"\n  > Seed: {seed}")
                result_dict = self._train_single_run(dataset_config, k_shot, seed, self.run_timestamp)
                k_shot_results.append(result_dict)

            # 完成当前k_shot的所有seed后，立即保存该k_shot的summary
            self._save_kshot_summary(dataset_name, k_shot, pretrained_results, k_shot_results)

    def _train_single_run(self, dataset_config, k_shot, seed, timestamp):
        """
        单次训练运行（指定数据集、K值、采样种子）

        Args:
            dataset: 数据集配置
            k_shot: K-shot数量
            seed: 用于数据采样的种子（每次不同，产生不同样本组合）
            timestamp: 时间戳
        """
        args = self.args
        dataset_name = dataset_config['name']
        run_output_dir = Path(args.output_dir) / timestamp / dataset_name / f'{k_shot}shot' / f'seed_{seed}'
        log_dir = run_output_dir / 'logs'
        vis_dir = run_output_dir / 'visualizations'
        log_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)
        # 创建logger
        self.logger = TrainingLogger(log_dir=log_dir, rank=self.rank, distributed=self.distributed)
        self.logger.log_config(self.args)
        self.logger.info(f"Dataset: {dataset_name}, K-shot: {k_shot}")
        self.logger.info(f"Sampling seed (for data selection): {seed}")
        # 创建或更新visualizer
        from lib.tools import Visualizer
        self.current_visualizer = Visualizer(output_dir=run_output_dir, rank=self.rank)

        # 重新加载预训练权重，确保每次训练独立
        self.logger.info("Reloading pretrained weights for independent training...")
        self._load_pretrained_weights()
        self._update_model_num_classes(dataset_config['num_classes'])
        self.build_optimizer()

        # 使用采样种子构建数据加载器（选择不同的样本组合）
        self.build_dataloader(dataset_config, k_shot, seed)
        self.logger.info(f"\n[Before Fine-tuning] Testing pretrained model on {dataset_name}...")


        # 设置全局训练种子以确保训练过程的可重复性
        from lib.utils import set_seed
        set_seed(self.global_train_seed)

        # 初始化训练历史
        loss_history = []
        early_stop = False

        for epoch in range(args.epochs):
            if early_stop:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break

            self.current_epoch = epoch
            # 调整学习率
            lr = self._adjust_learning_rate(epoch)
            # 训练（包含训练集的指标计算）
            train_metrics = self.train_epoch(epoch)
            train_metrics['lr'] = lr
            # 记录
            self.logger.log_epoch(epoch, train_metrics)

            # 记录loss历史
            loss_history.append(train_metrics['loss'])

            # 早停检查（需要至少 patience + 11 个epoch）
            if self.early_stop_enabled and epoch >= (self.early_stop_patience + 10):
                # 使用滑动平均平滑损失曲线
                loss_smooth = np.convolve(loss_history, np.ones(10)/10, mode='valid')

                # 计算阈值：旧损失的 threshold 比例
                threshold = self.early_stop_threshold * loss_smooth[-self.early_stop_patience]

                # 检查改进是否小于阈值
                improvement = loss_smooth[-self.early_stop_patience] - loss_smooth[-1]

                if improvement < threshold:
                    self.logger.info(f"Training loss plateau detected:")
                    self.logger.info(f"  Current loss: {loss_smooth[-1]:.5f}")
                    self.logger.info(f"  {self.early_stop_patience} epochs ago: {loss_smooth[-self.early_stop_patience]:.5f}")
                    self.logger.info(f"  Improvement: {improvement:.5f} < Threshold: {threshold:.5f}")
                    early_stop = True
         # 训练完成后绘制训练曲线
        self.current_visualizer.plot_training_curves(
            history=self.logger.history,
            dataset_name=dataset_name,
            k_shot=k_shot,
            sampling_seed=seed,
            logger=self.logger
        )
        self.logger.info(f"\n[After Fine-tuning] Testing final model on {dataset_name}...")
        after_metrics = self.test()
        self.logger.info(f"After Fine-tuning - Dice: {after_metrics['dice']:.4f}, IoU: {after_metrics['iou']:.4f}, HD95: {after_metrics['hd95']:.4f}")
        result_file = run_output_dir / 'test_results.json'
        result_dict = {
        'dataset': dataset_name,
        'k_shot': k_shot,
        'sampling_seed': seed,
        'test_dice': float(after_metrics['dice']),
        'test_iou': float(after_metrics['iou']),
        'test_hd95': float(after_metrics['hd95'])
        }
        with open(result_file, 'w') as f:
            json.dump(result_dict, f, indent=2)

        return result_dict

    def _save_kshot_summary(self, dataset_name, k_shot, pretrained_results, k_shot_results):
        """
        计算并保存K-shot的平均结果

        Args:
            dataset_name: 数据集名称
            k_shot: K-shot值
            k_shot_results: 所有种子的测试结果列表
        """
        args = self.args
        # 计算微调后的平均值和标准差（保持向后兼容）
        dice_scores = [r['test_dice'] for r in k_shot_results]
        iou_scores = [r['test_iou'] for r in k_shot_results]
        hd95_scores = [r['test_hd95'] for r in k_shot_results]
        summary = {
           'dataset': dataset_name,
            'k_shot': k_shot,
            'num_seeds': len(k_shot_results),
            'sampling_seeds': [r['sampling_seed'] for r in k_shot_results],
            'before_finetune': pretrained_results,
            # 微调后的结果
            'after_finetune': {
                'avg_dice': float(np.mean(dice_scores)),
                'std_dice': float(np.std(dice_scores)),
                'avg_iou': float(np.mean(iou_scores)),
                'std_iou': float(np.std(iou_scores)),
                'avg_hd95': float(np.mean(hd95_scores)),
                'std_hd95': float(np.std(hd95_scores))
            }
        }
        # 保存到k-shot目录
        kshot_dir = Path(args.output_dir) / self.run_timestamp / dataset_name / f'{k_shot}shot'
        kshot_dir.mkdir(parents=True, exist_ok=True)
        summary_file = kshot_dir / 'summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

    def _update_model_num_classes(self, num_classes):
        """更新模型输出类别数"""
        if hasattr(self.model, 'out_channels'):
            self.model.out_channels = num_classes
        self.wrap_model()

    def _adjust_learning_rate(self, epoch):
        """调整学习率（Warmup + Cosine）"""
        args = self.args
        init_lr = args.lr
        warmup_epochs = args.finetune_warmup_epochs
        total_epochs = args.epochs

        if epoch < warmup_epochs:
            lr = init_lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            lr = init_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr
