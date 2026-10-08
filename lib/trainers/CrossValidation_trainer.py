"""
交叉验证训练器

支持4种PEFT方法×3个权重×交叉验证的完整实验框架
执行流程：划分折→遍历折→加载测试数据→遍历权重→遍历方法→遍历shot×seed
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import math
import csv
from datetime import datetime

import lib.models as models
from lib.data.data_datasets import get_dataloader
from lib.data.cross_validation import split_kfold_cv, generate_cv_seed
from lib.tools.cv_result_manager import CVResultManager
from lib.tools.result_table_generator import ResultTableGenerator
from lib.tools import TrainingLogger, MetricsCalculator
from lib.utils import SmoothedValue, set_seed
from .base_trainer import BaseTrainer
from monai.inferers import sliding_window_inference


class CrossValidation_trainer(BaseTrainer):
    """
    交叉验证训练器（继承BaseTrainer）

    单次运行处理一个中心的所有方法×所有权重×所有折
    """

    def __init__(self, args):
        # 实验模式：
        # - legacy: 现有K折交叉验证（按current_center）
        # - kshot_tcga: 对target_centers逐个中心进行：目标中心k-shot + 全量TCGA训练，目标中心剩余样本测试
        self.experiment_mode = getattr(args, 'experiment_mode', 'legacy')

        # 先读取必要配置
        self.current_center = args.current_center
        self.peft_methods = args.peft_methods
        self.weight_paths = args.weight_paths
        self.cross_validation = args.cross_validation
        self.center_configs = args.center_configs

        # kshot_tcga 模式可遍历多个中心：优先用target_centers的第一个作为初始化占位
        target_centers = getattr(args, 'target_centers', None)
        if self.experiment_mode == 'kshot_tcga' and target_centers:
            self.current_center = list(target_centers)[0]

        self.center_config = self.center_configs[self.current_center]

        # 读取k_shot和随机种子配置
        k_shot_config = getattr(args, 'k_shot', [5, 10])
        if isinstance(k_shot_config, (list, tuple)) or hasattr(k_shot_config, '__iter__') and not isinstance(k_shot_config, str):
            self.k_shot_list = list(k_shot_config)
        else:
            self.k_shot_list = [k_shot_config]

        random_seeds_config = getattr(args, 'random_seeds', [0, 1, 2])
        self.random_seeds = list(random_seeds_config) if hasattr(random_seeds_config, '__iter__') else [random_seeds_config]

        self.global_train_seed = getattr(args, 'seed', 42)

        # 早停配置
        self.early_stop_patience = getattr(args, 'early_stop_patience', 10)
        self.early_stop_threshold = getattr(args, 'early_stop_threshold', 0.01)
        self.early_stop_enabled = getattr(args, 'early_stop_enabled', True)

        # 调用BaseTrainer初始化
        super().__init__(args)
        self.args = args

        # 重写学习率初始化（对于小batch_size，不使用batch_size缩放）
        # BaseTrainer的init_lr会缩放：lr * batch_size / 256
        # 对于batch_size=2，这会导致学习率过小（0.0005 * 2 / 256 ≈ 0.000004）
        # 因此直接使用配置的学习率，不进行batch_size缩放
        self.lr = args.lr

        # 初始化工具
        self.result_manager = CVResultManager()
        self.table_generator = ResultTableGenerator()
        self.metrics_calc = MetricsCalculator()

        print(f"\n{'='*80}")
        print(f"CrossValidation Trainer Initialized")
        print(f"{'='*80}")
        print(f"Center: {self.current_center}")
        print(f"Methods: {[m['name'] for m in self.peft_methods]}")
        print(f"Weights: {list(self.weight_paths.keys())}")
        if self.experiment_mode == 'legacy':
            print(f"Cross-validation folds: {self.cross_validation[self.current_center]}")
        else:
            print(f"Experiment mode: {self.experiment_mode}")
        print(f"{'='*80}\n")

    def build_model(self):
        """
        构建模型（占位模型）

        注意：CrossValidation_trainer在运行时会根据不同的方法和权重重新构建模型
        这里创建一个占位模型以满足BaseTrainer的要求
        """
        args = self.args

        # 使用第一个PEFT方法创建占位模型
        first_method = self.peft_methods[0]
        model_name = first_method['model_name']

        model_class = getattr(models, model_name, None)
        if model_class is None:
            raise ValueError(f"Model {model_name} not found in lib.models")

        # 创建占位模型
        self.model = model_class(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=self.center_config['num_classes'],
            feature_size=args.feature_size,
        )

        # 包装模型（不加载权重，权重在运行时加载）
        self.wrap_model()

        # 保存当前模型信息（用于后续重置）
        self.current_model_name = model_name
        self.current_weight_path = None  # 将在运行时设置

    def _load_pretrained_weights(self, weight_path=None):
        """
        加载预训练权重（不加载输出层，输出层随机初始化）
        """
        if weight_path is None:
            weight_path = getattr(self.args, 'pretrained_path', None)

        if not weight_path or not os.path.exists(weight_path):
            msg = f"Pretrained model not found at {weight_path}"
            print(f"Warning: {msg}") if not self.logger else self.logger.warning(msg)
            return

        # 加载checkpoint
        checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)

        # 处理命名并过滤输出层（不加载输出层权重）
        new_state_dict = {}

        for k, v in state_dict.items():
            key = k[7:] if k.startswith('module.') else k
            if not key.startswith('model.'):
                key = 'model.' + key

            # 跳过输出层权重，让输出层随机初始化
            if 'out.conv' in key or 'classifier' in key:
                continue

            new_state_dict[key] = v

        # 加载backbone和decoder（不包括输出层）
        incompatible_keys = self.model.load_state_dict(new_state_dict, strict=False)

        # 过滤并报告警告（排除输出层）
        filtered_missing = [k for k in incompatible_keys.missing_keys if 'out.conv' not in k and 'classifier' not in k]
        filtered_unexpected = [k for k in incompatible_keys.unexpected_keys if 'out.conv' not in k and 'classifier' not in k]

        if self.logger:
            self.logger.info(f"Loaded weights from {weight_path}")
            if filtered_missing:
                self.logger.warning(f"Missing keys: {len(filtered_missing)} (excluding output layer)")
            if filtered_unexpected:
                self.logger.warning(f"Unexpected keys: {len(filtered_unexpected)} (excluding output layer)")
        else:
            if filtered_missing or filtered_unexpected:
                print(f"Loaded weights: {len(filtered_missing)} missing, {len(filtered_unexpected)} unexpected (excluding output layer)")

    def run(self):
        """
        主流程

        执行流程：
        1. 准备交叉验证划分（只做1次）
        2. 遍历折→遍历权重→遍历方法→遍历shot×seed
        3. 保存所有cv_summary.json
        4. 生成汇总表格
        """
        if self.experiment_mode == 'kshot_tcga':
            self._run_kshot_tcga()
            return

        # legacy：时间戳格式：{timestamp}_{center}，便于识别不同中心的运行结果
        self.run_timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.current_center}"

        print(f"\n{'='*80}")
        print(f"Starting Cross-Validation Training")
        print(f"Timestamp: {self.run_timestamp}")
        print(f"{'='*80}\n")

        # 步骤1：准备交叉验证划分
        cv_splits = self._prepare_cv_splits()

        # 步骤2：执行训练
        all_results = self._execute_cv_training(cv_splits)

        # 步骤3：保存所有cv_summary.json
        self._save_all_cv_summaries(all_results)

        # 步骤4：生成汇总表格
        self._generate_result_table()

        print(f"\n{'='*80}")
        print(f"Cross-Validation Training Completed!")
        print(f"{'='*80}\n")

    def _run_kshot_tcga(self):
        """kshot+TCGA训练：逐个目标中心训练与评估（只测目标中心剩余样本）。"""
        target_centers = getattr(self.args, 'target_centers', None)
        if not target_centers:
            target_centers = [self.current_center]
        else:
            target_centers = list(target_centers)

        use_tcga = bool(getattr(self.args, 'use_tcga', True))
        tcga_center_name = getattr(self.args, 'tcga_center_name', 'TCGA')

        tcga_samples_abs = []
        if use_tcga:
            if tcga_center_name not in self.center_configs:
                raise KeyError(
                    f"TCGA center '{tcga_center_name}' not found in center_configs. "
                    f"Please add center_configs.{tcga_center_name} with csv_path/data_path."
                )
            tcga_cfg = self.center_configs[tcga_center_name]
            raw_tcga = self._load_samples_from_center_config(tcga_cfg)
            tcga_samples_abs = self._normalize_samples_to_abs(raw_tcga, tcga_cfg['data_path'])
            print(f"Loaded TCGA samples: {len(tcga_samples_abs)}")

        for center_name in target_centers:
            if center_name not in self.center_configs:
                raise KeyError(f"Target center '{center_name}' not found in center_configs")

            self.current_center = center_name
            self.center_config = self.center_configs[center_name]

            # 每个中心单独一个timestamp目录，避免混在一起
            self.run_timestamp = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{center_name}_kshot_tcga"

            print(f"\n{'='*80}")
            print(f"Starting kshot+TCGA Training")
            print(f"Center: {center_name}")
            print(f"Timestamp: {self.run_timestamp}")
            print(f"{'='*80}\n")

            # 目标中心全量样本（用于k-shot抽样 + 剩余测试）
            raw_center = self._load_samples_from_center_config(self.center_config)
            center_samples_abs = self._normalize_samples_to_abs(raw_center, self.center_config['data_path'])
            print(f"Loaded target-center samples: {len(center_samples_abs)}")

            # 为每个(k_shot, seed)预先构建一次train/test loader（不同weight/method复用）
            loader_bank = {}
            for k_shot in self.k_shot_list:
                for seed in self.random_seeds:
                    selected, remaining = self._split_kshot_remaining(
                        center_samples_abs, k_shot=k_shot, seed=seed
                    )

                    train_samples = list(selected)
                    if use_tcga:
                        train_samples.extend(tcga_samples_abs)

                    test_samples = list(remaining)

                    self.current_fold_idx = 0
                    train_loader = self._create_train_dataloader_from_samples(
                        train_samples,
                        context_override=(
                            f"{self.run_timestamp} | center={self.current_center} | mode={self.experiment_mode} | "
                            f"split=train | fold=0 | kshot={k_shot} | seed={seed} | tcga={'on' if use_tcga else 'off'}"
                        ),
                    )
                    test_loader = self._create_test_dataloader(
                        test_samples,
                        context_override=(
                            f"{self.run_timestamp} | center={self.current_center} | mode={self.experiment_mode} | "
                            f"split=test | fold=0 | kshot={k_shot} | seed={seed}"
                        ),
                    )
                    loader_bank[(k_shot, seed)] = (train_loader, test_loader)
                    print(
                        f"Prepared loaders for {k_shot}-shot, seed {seed}: "
                        f"train={len(train_loader.dataset)}, test={len(test_loader.dataset)}"
                    )

            # 初始化结果存储
            all_results = {}
            for method_config in self.peft_methods:
                method_name = method_config['name']
                all_results[method_name] = {w: [] for w in self.weight_paths}

            # 遍历权重/方法
            for weight_name, weight_path in self.weight_paths.items():
                print(f"\n>>> Weight: {weight_name}")
                for method_config in self.peft_methods:
                    method_name = method_config['name']
                    model_name = method_config['model_name']
                    print(f"  >> Method: {method_name}")

                    self._build_model_with_weight(model_name, weight_path)

                    # 训练所有shot×seed组合（单fold：fold_idx=0）
                    for k_shot in self.k_shot_list:
                        for seed in self.random_seeds:
                            print(f"      {k_shot}-shot, seed {seed}")

                            train_loader, test_loader = loader_bank[(k_shot, seed)]

                            result = self._train_single_run(
                                train_loader,
                                test_loader,
                                0,
                                method_name,
                                weight_name,
                                k_shot,
                                seed,
                            )
                            if result is not None:
                                all_results[method_name][weight_name].append(result)
                            else:
                                print(f"        ⚠️  Skipped invalid result for {k_shot}-shot, seed {seed}")

            # 释放预构建的loader
            for train_loader, test_loader in loader_bank.values():
                del train_loader
                del test_loader
            torch.cuda.empty_cache()

            # 单fold汇总（n_folds=1）
            self._save_all_cv_summaries(all_results, n_folds_override=1)
            self._generate_result_table()

            print(f"\n{'='*80}")
            print(f"kshot+TCGA Training Completed for {center_name}!")
            print(f"{'='*80}\n")

    def _load_samples_from_center_config(self, center_cfg):
        """从center_config读取全部样本（兼容csv_path或finetune_csv+test_csv）。"""
        samples = []
        csv_path = center_cfg.get('csv_path')
        if csv_path:
            with open(csv_path) as f:
                samples.extend(list(csv.DictReader(f)))
        else:
            with open(center_cfg['finetune_csv']) as f:
                samples.extend(list(csv.DictReader(f)))
            with open(center_cfg['test_csv']) as f:
                samples.extend(list(csv.DictReader(f)))
        return samples

    def _normalize_samples_to_abs(self, samples, data_path):
        """把sample list中的image/label路径转换为绝对路径，便于跨中心拼接。"""
        root = Path(data_path)
        normalized = []
        for row in samples:
            new_row = dict(row)
            for key in ['image', 'label']:
                if key in new_row and new_row[key] is not None:
                    p = str(new_row[key]).strip()
                    new_row[key] = p if os.path.isabs(p) else str(root / p)
            normalized.append(new_row)
        return normalized

    def _split_kshot_remaining(self, samples_abs, k_shot, seed):
        """对目标中心样本做k-shot抽样，返回(selected, remaining)。"""
        if not samples_abs:
            return [], []
        rng = random.Random(seed)
        indices = list(range(len(samples_abs)))
        rng.shuffle(indices)
        k = min(int(k_shot), len(indices))
        selected_idx = set(indices[:k])
        selected = [samples_abs[i] for i in range(len(samples_abs)) if i in selected_idx]
        remaining = [samples_abs[i] for i in range(len(samples_abs)) if i not in selected_idx]
        return selected, remaining

    def _create_train_dataloader_from_samples(self, train_samples, context_override=None):
        """用给定样本列表直接创建训练DataLoader（不再内部random.sample）。"""
        context = context_override or (
            f"{self.run_timestamp} | center={self.current_center} | mode={self.experiment_mode} | "
            f"split=train | fold={getattr(self, 'current_fold_idx', 'NA')}"
        )
        return get_dataloader(
            data_path=self.center_config['data_path'],
            list_file=None,
            sample_list=train_samples,
            batch_size=self.args.batch_size,
            num_workers=self.workers,
            is_train=True,
            distributed=self.distributed,
            args=self.args,
            use_preload=True,
            context=context,
        )

    def _prepare_cv_splits(self):
        """
        准备交叉验证划分（只做1次，与方法无关）

        Returns:
            List[(train_samples, test_samples)]: CV折列表
        """
        print("Preparing cross-validation splits...")

        # 读取所有样本：优先使用单一csv_path，兼容旧的finetune_csv+test_csv
        all_samples = []
        csv_path = self.center_config.get('csv_path')
        if csv_path:
            with open(csv_path) as f:
                all_samples.extend(list(csv.DictReader(f)))
        else:
            with open(self.center_config['finetune_csv']) as f:
                all_samples.extend(list(csv.DictReader(f)))
            with open(self.center_config['test_csv']) as f:
                all_samples.extend(list(csv.DictReader(f)))

        print(f"  Total samples: {len(all_samples)}")

        # 划分
        n_folds = self.cross_validation[self.current_center]
        cv_seed = generate_cv_seed(self.current_center)
        cv_folds = split_kfold_cv(all_samples, n_folds, cv_seed)

        print(f"  Split into {n_folds} folds (CV seed: {cv_seed})")
        for i, (train, test) in enumerate(cv_folds):
            print(f"    Fold {i}: {len(train)} train, {len(test)} test")

        return cv_folds

    def _execute_cv_training(self, cv_splits):
        """
        执行交叉验证训练

        流程：遍历折→遍历权重→遍历方法→遍历shot×seed

        Args:
            cv_splits: CV划分结果

        Returns:
            all_results: {method_name: {weight_name: [results]}}
        """
        # 初始化结果存储
        all_results = {}
        for method_config in self.peft_methods:
            method_name = method_config['name']
            all_results[method_name] = {w: [] for w in self.weight_paths}

        # 遍历折
        for fold_idx, (fold_train, fold_test) in enumerate(cv_splits):
            self.current_fold_idx = fold_idx
            print(f"\n{'='*80}")
            print(f"Fold {fold_idx+1}/{len(cv_splits)}")
            print(f"Train: {len(fold_train)} samples, Test: {len(fold_test)} samples")
            print(f"{'='*80}")

            # 加载测试数据（所有权重和方法共享）
            print("Loading test data (shared across all weights and methods)...")
            test_loader = self._create_test_dataloader(
                fold_test,
                context_override=(
                    f"{self.run_timestamp} | center={self.current_center} | mode={self.experiment_mode} | "
                    f"split=test | fold={fold_idx}"
                ),
            )
            print(f"  Loaded {len(test_loader.dataset)} test samples")

            # 遍历权重
            for weight_name, weight_path in self.weight_paths.items():
                print(f"\n>>> Weight: {weight_name}")

                # 遍历方法
                for method_config in self.peft_methods:
                    method_name = method_config['name']
                    model_name = method_config['model_name']
                    print(f"  >> Method: {method_name}")

                    # 构建模型并加载权重
                    self._build_model_with_weight(model_name, weight_path)

                    # 训练所有shot×seed组合
                    results = self._train_method_all_shots(
                        fold_train, test_loader,
                        fold_idx, method_name, weight_name
                    )
                    all_results[method_name][weight_name].extend(results)

            # 释放测试数据
            del test_loader
            torch.cuda.empty_cache()
            print("Test data released")

        return all_results

    def _build_model_with_weight(self, model_name, weight_path):
        """
        构建模型并加载权重

        Args:
            model_name: 模型类名
            weight_path: 权重文件路径
        """
        # 如果模型类型改变，需要重新创建模型
        if not hasattr(self, 'current_model_name') or self.current_model_name != model_name:
            # 删除旧模型
            if self.model is not None:
                del self.model
                if hasattr(self, 'wrapped_model') and self.wrapped_model is not None:
                    del self.wrapped_model
                torch.cuda.empty_cache()

            # 创建新模型
            model_class = getattr(models, model_name, None)
            if model_class is None:
                raise ValueError(f"Model {model_name} not found in lib.models")

            self.model = model_class(
                img_size=(self.args.roi_x, self.args.roi_y, self.args.roi_z),
                in_channels=self.args.in_channels,
                out_channels=self.center_config['num_classes'],
                feature_size=self.args.feature_size,
            )

            # 包装模型
            self.wrap_model()

            # 更新当前模型名称
            self.current_model_name = model_name

        # 加载权重
        self._load_pretrained_weights(weight_path)
        self.current_weight_path = weight_path  # 保存用于后续重置

        # 构建优化器
        self.build_optimizer()

    def _train_method_all_shots(self, fold_train, test_loader,
                                fold_idx, method_name, weight_name):
        """
        训练所有shot×seed组合

        Args:
            fold_train: 训练样本列表
            test_loader: 测试数据加载器
            fold_idx: 折索引
            method_name: 方法名称
            weight_name: 权重名称

        Returns:
            List[dict]: 所有结果
        """
        results = []

        for k_shot in self.k_shot_list:
            for seed in self.random_seeds:
                print(f"      {k_shot}-shot, seed {seed}")

                # 加载K个训练样本
                train_loader = self._create_kshot_dataloader(fold_train, k_shot, seed)
                print(f"        Loaded {len(train_loader.dataset)} training samples")

                # 训练单次
                result = self._train_single_run(
                    train_loader, test_loader,
                    fold_idx, method_name, weight_name, k_shot, seed
                )
                # 只保存有效的结果（如果出错，result为None，不保存）
                if result is not None:
                    results.append(result)
                else:
                    print(f"        ⚠️  Skipped invalid result for {k_shot}-shot, seed {seed}")

                # 释放训练数据
                del train_loader
                torch.cuda.empty_cache()
                print(f"        Training data released")

        return results

    def _train_single_run(self, train_loader, test_loader,
                         fold_idx, method_name, weight_name, k_shot, seed):
        """
        单次训练

        关键：确保模型状态正确重置，无梯度累积
        """
        # Restore the pretrained model state before each run.
        self._reset_model_to_pretrained()

        # Clear gradients from the previous run.
        self.optimizer.zero_grad()
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

        # Set the global training seed.
        set_seed(self.global_train_seed)

        # 步骤4：设置logger
        output_dir = self._get_output_dir(method_name, weight_name, fold_idx, k_shot, seed)
        log_dir = output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = TrainingLogger(log_dir=log_dir, rank=self.rank, distributed=self.distributed)

        # 设置train_loader
        self.train_loader = train_loader

        # 步骤5：训练
        loss_history = []
        early_stop = False
        epochs_trained = 0  # 初始化，避免作用域问题

        try:
            for epoch in range(self.args.epochs):
                epochs_trained = epoch + 1  # 记录实际训练的epoch数

                if early_stop:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

                self.current_epoch = epoch

                # 调整学习率
                lr = self._adjust_learning_rate(epoch)

                # 训练一个epoch
                train_metrics = self.train_epoch(epoch)
                train_metrics['lr'] = lr

                # 记录
                if self.logger:
                    self.logger.log_epoch(epoch, train_metrics)

                # 记录loss历史
                loss_history.append(train_metrics['loss'])

                # 早停检查（确保有足够的历史数据）
                if self.early_stop_enabled and len(loss_history) >= (self.early_stop_patience + 10):
                    loss_smooth = np.convolve(loss_history, np.ones(10)/10, mode='valid')
                    if len(loss_smooth) >= self.early_stop_patience:
                        threshold = self.early_stop_threshold * loss_smooth[-self.early_stop_patience]
                        improvement = loss_smooth[-self.early_stop_patience] - loss_smooth[-1]

                        # Stop when smoothed loss improvement is below the relative threshold.
                        # This includes small loss decreases and loss increases.
                        if improvement < threshold:
                            if self.logger:
                                self.logger.info(f"Training loss plateau detected")
                                self.logger.info(f"  Current loss: {loss_smooth[-1]:.5f}")
                                self.logger.info(f"  {self.early_stop_patience} epochs ago: {loss_smooth[-self.early_stop_patience]:.5f}")
                                self.logger.info(f"  Improvement: {improvement:.5f} < Threshold: {threshold:.5f}")
                            early_stop = True
        except Exception as e:
            error_msg = f"Training failed for {method_name}/{weight_name}/fold_{fold_idx}/{k_shot}shot/seed_{seed}: {e}"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(f"ERROR: {error_msg}")
            # 训练失败，不保存结果
            return None

        # 步骤6：测试
        try:
            test_metrics = self.test(test_loader, save_dir=output_dir)
        except Exception as e:
            error_msg = f"Test failed for {method_name}/{weight_name}/fold_{fold_idx}/{k_shot}shot/seed_{seed}: {e}"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(f"ERROR: {error_msg}")
            # 不保存错误的结果
            return None

        # 检查测试是否有效（至少有一些有效样本）
        if test_metrics.get('total_test_samples', 0) > 0:
            valid_samples = test_metrics.get('total_test_samples', 0) - test_metrics.get('samples_skipped', 0)
            if valid_samples == 0:
                error_msg = f"All test samples were skipped for {method_name}/{weight_name}/fold_{fold_idx}/{k_shot}shot/seed_{seed}"
                if self.logger:
                    self.logger.error(error_msg)
                else:
                    print(f"ERROR: {error_msg}")
                # 不保存无效的结果
                return None

        # 步骤7：记录结果
        result = {
            'peft_method': method_name,
            'center': self.current_center,
            'weight': weight_name,
            'fold': fold_idx,
            'k_shot': k_shot,
            'seed': seed,
            'test_dice': float(test_metrics['dice']),
            'test_iou': float(test_metrics['iou']),
            'test_hd95': float(test_metrics['hd95']),
            'epochs_trained': epochs_trained,
            'early_stopped': early_stop,
            'samples_skipped': test_metrics.get('samples_skipped', 0),
            'total_test_samples': test_metrics.get('total_test_samples', 0)
        }

        # 验证并保存结果（如果验证失败，会抛出异常，不会保存）
        try:
            self.result_manager.validate_result(result)
            self.result_manager.save_single_result(result, output_dir)
        except AssertionError as e:
            error_msg = f"Result validation failed for {method_name}/{weight_name}/fold_{fold_idx}/{k_shot}shot/seed_{seed}: {e}"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(f"ERROR: {error_msg}")
            # 验证失败，不保存结果
            return None

        print(f"        Result: Dice={result['test_dice']:.4f}, "
              f"IoU={result['test_iou']:.4f}, HD95={result['test_hd95']:.2f}")

        return result

    def _reset_model_to_pretrained(self):
        """
        重置模型到预训练状态

        关键：确保每次训练独立，无梯度累积
        """
        # 保存LoRA应用状态（用于检测结构变化）
        lora_was_applied = getattr(self.model, '_lora_applied', False) if hasattr(self, 'model') and self.model is not None else False

        # 重新加载预训练权重
        self._load_pretrained_weights(self.current_weight_path)

        # Rewrap the model if devices differ or reloading changes the LoRA structure.
        if hasattr(self, 'wrapped_model') and self.model is not None:
            try:
                model_device = next(self.model.parameters()).device
                wrapped_device = next(self.wrapped_model.parameters()).device

                # 检查LoRA状态是否改变（重置或重新应用）
                lora_now_applied = getattr(self.model, '_lora_applied', False)
                lora_structure_changed = (lora_was_applied != lora_now_applied)

                # 如果设备不匹配，或者LoRA结构已改变，重新包装
                if model_device != wrapped_device:
                    if self.logger:
                        self.logger.warning(f"设备不匹配，重新包装模型 (model: {model_device}, wrapped: {wrapped_device})")
                    self.wrap_model()
                elif lora_structure_changed:
                    # LoRA结构已改变（重置或重新应用），需要重新包装以确保wrapped_model指向正确的结构
                    if self.logger:
                        self.logger.info(f"LoRA结构已改变 (was_applied={lora_was_applied}, now_applied={lora_now_applied})，重新包装以确保结构一致")
                    self.wrap_model()
            except (StopIteration, RuntimeError) as e:
                # 如果无法获取设备信息，重新包装以确保安全
                if self.logger:
                    self.logger.warning(f"无法检查设备一致性，重新包装模型: {e}")
                self.wrap_model()
        else:
            # 如果没有wrapped_model，需要包装
            if self.model is not None:
                self.wrap_model()

        # 重新创建优化器（这会自动清除旧的state，并基于当前模型参数创建新的优化器）
        # 注意：必须在模型权重加载完成后调用，确保优化器指向正确的参数
        self.build_optimizer()

        # 清除CUDA缓存
        torch.cuda.empty_cache()

    def _create_test_dataloader(self, test_samples, context_override=None):
        """
        创建测试数据加载器

        Args:
            test_samples: 测试样本列表（CSV行字典）

        Returns:
            DataLoader
        """
        context = context_override or (
            f"{self.run_timestamp} | center={self.current_center} | mode={self.experiment_mode} | "
            f"split=test | fold={getattr(self, 'current_fold_idx', 'NA')}"
        )
        return get_dataloader(
            data_path=self.center_config['data_path'],
            list_file=None,
            sample_list=test_samples,
            batch_size=1,
            num_workers=self.workers,
            is_train=False,
            distributed=False,
            args=self.args,
            use_preload=True,
            context=context,
        )

    def _create_kshot_dataloader(self, train_samples, k_shot, seed):
        """
        创建K-shot训练数据加载器

        Args:
            train_samples: 训练样本列表
            k_shot: K值
            seed: 随机种子

        Returns:
            DataLoader
        """
        # 采样K个样本
        random.seed(seed)
        selected = random.sample(train_samples, min(k_shot, len(train_samples)))

        # 创建DataLoader
        context = (
            f"{self.run_timestamp} | center={self.current_center} | mode={self.experiment_mode} | "
            f"split=train | fold={getattr(self, 'current_fold_idx', 'NA')} | kshot={k_shot} | seed={seed}"
        )
        return get_dataloader(
            data_path=self.center_config['data_path'],
            list_file=None,
            sample_list=selected,
            batch_size=self.args.batch_size,
            num_workers=self.workers,
            is_train=True,
            distributed=self.distributed,
            args=self.args,
            use_preload=True,
            context=context,
        )

    def _get_output_dir(self, method_name, weight_name, fold_idx, k_shot, seed):
        """
        生成输出目录路径

        格式: outputs/{timestamp}_{center}/{method}/{weight}/
              fold_{fold_idx}/{k_shot}shot/seed_{seed}/

        注意：时间戳已包含中心名称，因此路径中不再需要单独的center层级
        """
        return Path(self.args.output_dir) / self.run_timestamp / \
               method_name / weight_name / \
               f'fold_{fold_idx}' / f'{k_shot}shot' / f'seed_{seed}'

    def _save_all_cv_summaries(self, all_results, n_folds_override=None):
        """
        保存所有cv_summary.json

        Args:
            all_results: {method_name: {weight_name: [results]}}
        """
        print(f"\n{'='*80}")
        print("Saving CV summaries...")
        print(f"{'='*80}")

        n_folds = n_folds_override if n_folds_override is not None else self.cross_validation[self.current_center]

        for method_name in all_results:
            for weight_name in all_results[method_name]:
                # 计算汇总
                summary = self.result_manager.compute_cv_summary(
                    all_results[method_name][weight_name],
                    method_name, self.current_center, weight_name,
                    n_folds, self.k_shot_list, seed_list=self.random_seeds
                )

                # 验证并保存
                if self.result_manager.validate_cv_summary(summary, n_folds):
                    # 时间戳已包含中心名称，路径中不再需要单独的center层级
                    output_path = Path(self.args.output_dir) / self.run_timestamp / \
                                 method_name / weight_name / 'cv_summary.json'
                    self.result_manager.save_cv_summary(summary, output_path)

    def _generate_result_table(self):
        """生成结果表格"""
        print(f"\n{'='*80}")
        print("Generating result tables...")
        print(f"{'='*80}")

        # 收集所有cv_summary.json
        summaries = self.table_generator.collect_cv_summaries(
            self.args.output_dir, self.run_timestamp
        )

        if not summaries:
            print("No summaries found, skipping table generation")
            return

        # 生成表格
        df = self.table_generator.generate_method_center_table(summaries)

        # 保存多种格式
        output_dir = Path(self.args.output_dir) / self.run_timestamp / 'summaries'
        self.table_generator.save_table(df, output_dir, 'by_method_center')

        # 打印预览
        self.table_generator.print_table_preview(df)

    # ========== 继承自BaseTrainer的方法 ==========

    def _compute_loss(self, outputs, labels):
        """计算分割损失（Dice Loss + CE Loss）"""
        dice_loss = self._dice_loss(outputs, labels)
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

            # 计算指标
            with torch.no_grad():
                dice = self.metrics_calc.dice_score(outputs, labels)
                iou = self.metrics_calc.iou_score(outputs, labels)

            loss_meter.update(loss.item())
            dice_meter.update(dice)
            iou_meter.update(iou)

            if self.logger and i % 5 == 0:
                self.logger.info(
                    f"Epoch: {epoch:03d} | Loss: {loss.item():.4f} | "
                    f"Dice: {dice:.4f} | IoU: {iou:.4f}"
                )

        return {
            'loss': loss_meter.global_avg,
            'dice': dice_meter.global_avg,
            'iou': iou_meter.global_avg,
        }

    def test(self, test_loader, save_dir=None):
        """
        测试模型

        Args:
            test_loader: 测试数据加载器

        Returns:
            dict: 测试指标
        """
        self.model.eval()
        self._qual_idx = 0
        dice_scores, iou_scores, hd95_scores = [], [], []

        roi_size = (self.args.roi_x, self.args.roi_y, self.args.roi_z)
        sw_batch_size = getattr(self.args, 'sw_batch_size', 4)
        overlap = getattr(self.args, 'infer_overlap', 0.5)

        skipped_count = 0
        total_samples = len(test_loader.dataset) if hasattr(test_loader, 'dataset') else 0

        with torch.no_grad():
            for batch in test_loader:
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

                if save_dir is not None and getattr(self.args, 'save_predictions', False):
                    import os as _os
                    _os.makedirs(save_dir, exist_ok=True)
                    _pred = torch.argmax(outputs, dim=1).cpu().numpy().astype('uint8')
                    _img = images.detach().cpu().numpy().astype('float16')
                    _lab = labels.detach().cpu().numpy().astype('uint8')
                    for _b in range(_pred.shape[0]):
                        np.savez_compressed(_os.path.join(save_dir, f'qual_case{self._qual_idx}.npz'), img=_img[_b], gt=_lab[_b], pred=_pred[_b])
                        self._qual_idx += 1

                # 计算指标
                dice = self.metrics_calc.dice_score(outputs, labels)

                # 仅在指标异常时跳过：dice=0 是有效结果，不应跳过
                if np.isnan(dice) or np.isinf(dice):
                    skipped_count += 1
                    if self.logger:
                        self.logger.warning(
                            f"Skipped sample with invalid dice={dice} (total skipped: {skipped_count})"
                        )
                    continue

                dice_scores.append(dice)
                iou_scores.append(self.metrics_calc.iou_score(outputs, labels))
                hd95_scores.append(self.metrics_calc.hausdorff_distance_95(outputs, labels))

        if skipped_count > 0:
            msg = f"Skipped {skipped_count}/{total_samples} samples with invalid dice (NaN/Inf)"
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)

        if len(dice_scores) == 0:
            if self.logger:
                self.logger.warning("All test samples were skipped (invalid dice: NaN/Inf)")
            else:
                print("Warning: All test samples were skipped (invalid dice: NaN/Inf)")
            return {
                'dice': 0.0,
                'iou': 0.0,
                'hd95': float('inf'),
                'samples_skipped': skipped_count,
                'total_test_samples': total_samples
            }

        return {
            'dice': np.mean(dice_scores),
            'iou': np.mean(iou_scores),
            'hd95': np.mean(hd95_scores),
            'samples_skipped': skipped_count,
            'total_test_samples': total_samples
        }

    def _adjust_learning_rate(self, epoch):
        """调整学习率（Warmup + Cosine）"""
        # 使用配置的学习率（不进行batch_size缩放，因为batch_size=2时缩放会导致学习率过小）
        init_lr = self.lr
        warmup_epochs = self.args.finetune_warmup_epochs
        total_epochs = self.args.epochs

        if epoch < warmup_epochs:
            lr = init_lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            lr = init_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr
