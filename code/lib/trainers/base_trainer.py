import os
import math
import torch
import torch.nn as nn
from pathlib import Path
import sys
sys.path.append('..')
from lib.tools import TrainingLogger, MetricsCalculator, Visualizer, CheckpointManager

class BaseTrainer(object):
    """
    基础训练器 - 只包含核心训练方法
    所有辅助功能都通过tools模块实现
    """

    def __init__(self, args):
        self.args = args
        self.rank = getattr(args, 'rank', 0)
        self.distributed = getattr(args, 'distributed', False)
        self.epochs = args.epochs
        # 核心组件
        self.model = None
        self.wrapped_model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None

        self.batch_size = args.batch_size
        self.workers = getattr(args, 'workers', 4)
        self.start_epoch = 0
        self.current_epoch = 0

        # 学习率初始化
        self.init_lr()

        # 工具模块（logger在子类中初始化）
        self.logger = None  # 将在子类中初始化
        self.metrics_calc = MetricsCalculator()
        self.visualizer = Visualizer(
            output_dir=args.output_dir,
            rank=self.rank
        )
        self.ckpt_manager = CheckpointManager(
            checkpoint_dir=Path(args.output_dir) / 'checkpoints',
            rank=self.rank,
            keep_last_n=3
        )

    def init_lr(self):
        """初始化学习率"""
        args = self.args
        # 智能获取学习率
        lr = args.lr
        # 线性缩放学习率
        self.lr = lr * self.batch_size / 256

    def build_model(self):
        """构建模型 - 子类需要实现"""
        raise NotImplementedError("build_model must be implemented by subclass")

    def build_optimizer(self):
        """构建优化器"""
        assert self.model is not None, "Model must be built before optimizer"

        args = self.args
        optim_params = self._get_optimizer_params()

        # 支持多种优化器
        if args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(
                optim_params,
                lr=self.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )
        elif args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                optim_params,
                lr=self.lr,
                weight_decay=args.weight_decay
            )
        elif args.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                optim_params,
                lr=self.lr,
                weight_decay=args.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {args.optimizer}")

        if self.logger:
            self.logger.info(f"Optimizer {args.optimizer} created")

    def _get_optimizer_params(self):
        """获取优化器参数组"""
        model = self.wrapped_model

        # 区分需要权重衰减和不需要权重衰减的参数
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # bias、LayerNorm、BatchNorm等不使用权重衰减
            if len(param.shape) == 1 or name.endswith('.bias'):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return [
            {'params': decay_params, 'weight_decay': self.args.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

    def build_dataloader(self):
        """构建数据加载器 - 子类需要实现"""
        raise NotImplementedError("build_dataloader must be implemented by subclass")
    def wrap_model(self):
        """
        统一的模型放置与（可选）DDP 包装，由 --device 决定：
        - cpu: 放在 CPU
        - cuda: 放在单 GPU
        - ddp: 转 SyncBN，DDP 包装
        """
        args = self.args
        self.device = args.device
        model = self.model
        assert model is not None, "Model must be built before wrapping"

        # 可训练参数检查
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise RuntimeError("Model has no trainable parameters")

        # 放设备
        if args.device == 'cpu':
            model = model.to('cpu')
            self.wrapped_model = model
        elif args.device == 'cuda':
            model = model.to(self.device)
            self.wrapped_model = model
        elif args.device == 'ddp':
            # 多卡 BN 同步 + 放设备 + DDP 包装
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = model.to(self.device)

            # 按卡数缩放 batch_size / workers（从 args 兜底）
            world_size = getattr(args, 'world_size', torch.cuda.device_count())
            original_bs = getattr(self, 'batch_size', getattr(args, 'batch_size', 32))
            original_workers = getattr(self, 'workers', getattr(args, 'workers', 4))
            self.batch_size = max(1, original_bs // max(1, world_size))
            self.workers = max(1, (original_workers + world_size - 1) // max(1, world_size))

            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.gpu],
                find_unused_parameters=True
            )
            self.wrapped_model = model
        else:
            raise ValueError("Unknown device mode")

        if self.logger:
            self.logger.info(f"Model wrapped on {args.device} successfully")

        # 输出参数统计
        self._print_parameter_stats()

    def _print_parameter_stats(self):
        """打印模型参数统计信息"""
        model = self.model

        # 统计总参数和可训练参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        # 格式化输出
        info_lines = [
            "\n" + "="*70,
            "Model Parameter Statistics",
            "="*70,
            f"Total parameters:        {total_params:,} ({total_params/1e6:.2f}M)",
            f"Trainable parameters:    {trainable_params:,} ({trainable_params/1e6:.2f}M)",
            f"Frozen parameters:       {frozen_params:,} ({frozen_params/1e6:.2f}M)",
            f"Trainable ratio:         {trainable_params/total_params*100:.2f}%",
            "="*70,
        ]

        # 输出到logger或print
        if self.logger:
            for line in info_lines:
                self.logger.info(line)
        else:
            for line in info_lines:
                print(line)

    def _load_pretrained_weights(self):
        """加载预训练权重（可重复调用，确保每次训练独立）"""
        args = self.args

        if not os.path.exists(args.pretrained_path):
            print(f"Warning: Pretrained model not found at {args.pretrained_path}")
            return

        checkpoint = torch.load(args.pretrained_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)

        # 处理state_dict命名
        new_state_dict = {}
        for k, v in state_dict.items():
            key = k[7:] if k.startswith('module.') else k
            if not key.startswith('model.'):
                key = 'model.' + key
            new_state_dict[key] = v

        # 加载权重并捕获未成功导入的键
        incompatible_keys = self.model.load_state_dict(new_state_dict, strict=False)
        # 使用logger输出权重加载情况
        if self.logger:
            self.logger.info(f"Loading pretrained weights from {args.pretrained_path}")

            if incompatible_keys.missing_keys:
                self.logger.warning(f"Missing keys in checkpoint ({len(incompatible_keys.missing_keys)} keys):")
                for key in incompatible_keys.missing_keys:
                    self.logger.warning(f"  - {key}")

            if incompatible_keys.unexpected_keys:
                self.logger.warning(f"Unexpected keys in checkpoint ({len(incompatible_keys.unexpected_keys)} keys):")
                for key in incompatible_keys.unexpected_keys:
                    self.logger.warning(f"  - {key}")

            if not incompatible_keys.missing_keys and not incompatible_keys.unexpected_keys:
                self.logger.info("All pretrained weights loaded successfully!")
        # 加载权重
        self.model.load_state_dict(new_state_dict, strict=False)

    def resume(self, checkpoint_path=None):
        """恢复训练"""
        if checkpoint_path is None:
            checkpoint_path = self.args.resume

        if not checkpoint_path or not os.path.exists(checkpoint_path):
            if self.logger:
                self.logger.info("No checkpoint to resume")
            return

        self.start_epoch = self.ckpt_manager.load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer
        )
        self.current_epoch = self.start_epoch

    def save_checkpoint(self, epoch, is_best=False, metric_value=None):
        """保存检查点"""
        arch = getattr(self.args, 'arch', None)
        if arch is None:
            arch = getattr(self.args, 'model_name', None)
        if arch is None and hasattr(self.model, '__class__'):
            arch = self.model.__class__.__name__

        state = {
            'epoch': epoch + 1,
            'arch': arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_metric': metric_value
        }

        self.ckpt_manager.save_checkpoint(
            state,
            epoch,
            is_best=is_best,
            metric_value=metric_value
        )

    def adjust_learning_rate(self, epoch):
        """
        调整学习率 - Cosine退火 + Warmup
        """
        args = self.args
        init_lr = self.lr

        # 智能获取训练参数
        warmup_epochs = getattr(
            self.args,
            'warmup_epochs',
            getattr(self.args, 'finetune_warmup_epochs', getattr(self.args, 'bridge_warmup_epochs', 10))
        )
        total_epochs = getattr(
            self.args,
            'epochs',
            getattr(self.args, 'finetune_epochs', getattr(self.args, 'bridge_epochs', 200))
        )

        if epoch < warmup_epochs:
            # Warmup阶段
            cur_lr = init_lr * (epoch + 1) / warmup_epochs
        else:
            # Cosine退火
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            cur_lr = init_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            if 'fix_lr' in param_group and param_group['fix_lr']:
                param_group['lr'] = init_lr
            else:
                param_group['lr'] = cur_lr

        return cur_lr

    def train_epoch(self, epoch):
        """训练一个epoch - 子类需要实现"""
        raise NotImplementedError("train_epoch must be implemented by subclass")

    def validate(self):
        """验证 - 子类可选实现"""
        return {}

    def run(self):
        """主训练循环"""
        args = self.args

        # 智能获取训练参数
        epochs = self.epochs
        save_freq = self.save_freq

        for epoch in range(self.start_epoch, epochs):
            self.current_epoch = epoch

            # 设置分布式sampler的epoch
            if args.distributed and hasattr(self.train_loader, 'sampler'):
                self.train_loader.sampler.set_epoch(epoch)

            # 调整学习率
            current_lr = self.adjust_learning_rate(epoch)

            # 训练一个epoch
            train_metrics = self.train_epoch(epoch)

            # 验证（如果有）
            if self.val_loader is not None:
                val_metrics = self.validate()
                train_metrics.update({f'val_{k}': v for k, v in val_metrics.items()})

            # 记录学习率
            train_metrics['lr'] = current_lr

            # 记录epoch结果
            self.logger.log_epoch(epoch, train_metrics)

            # 保存检查点
            if (epoch + 1) % save_freq == 0:
                is_best = self._is_best_model(train_metrics)
                metric_value = train_metrics.get('val_dice', train_metrics.get('dice', 0.0))
                self.save_checkpoint(epoch, is_best=is_best, metric_value=metric_value)

        # 保存训练曲线
        self.visualizer.save_training_curves(
            self.logger.history,
            stage_name=getattr(self, 'stage_name', 'training')
        )

    def _is_best_model(self, metrics):
        """判断是否是最佳模型"""
        # 子类可以重写这个方法
        metric_key = 'val_dice' if 'val_dice' in metrics else 'dice'
        if metric_key not in metrics:
            return False

        current_metric = metrics[metric_key]
        if self.ckpt_manager.best_metric is None:
            return True

        return current_metric > self.ckpt_manager.best_metric
