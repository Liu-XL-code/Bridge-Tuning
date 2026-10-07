import os
import torch
import shutil
from pathlib import Path


class CheckpointManager:
    """模型检查点管理工具"""

    def __init__(self, checkpoint_dir, rank=0, keep_last_n=3, early_stopping_patience=None, min_delta=0.001):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.is_main_process = (rank == 0)
        self.keep_last_n = keep_last_n
        self.best_metric = None

        # 早停机制
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta
        self.patience_counter = 0
        self.should_stop = False

    def save_checkpoint(self, state, epoch, is_best=False, metric_value=None):
        """
        保存检查点（仅保存最优模型）
        Args:
            state: 包含模型、优化器等状态的字典
            epoch: 当前epoch
            is_best: 是否是最佳模型
            metric_value: 评价指标值
        """
        if not self.is_main_process:
            return

        # 如果是最佳模型，保存最佳检查点
        if is_best:
            best_path = self.checkpoint_dir / 'best_checkpoint.pth'
            torch.save(state, best_path)
            self.best_metric = metric_value
            # 重置早停计数器
            self.patience_counter = 0
            print(f"Saved best checkpoint at epoch {epoch} with metric: {metric_value:.4f}")
        else:
            # 更新早停计数器
            if self.early_stopping_patience is not None:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    self.should_stop = True
                    print(f"Early stopping triggered after {self.patience_counter} epochs without improvement")

    def _cleanup_checkpoints(self):
        """清理旧的检查点文件"""
        checkpoints = sorted(
            self.checkpoint_dir.glob('checkpoint_epoch_*.pth'),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        # 保留最近的keep_last_n个检查点
        for checkpoint in checkpoints[self.keep_last_n:]:
            checkpoint.unlink()

    def load_checkpoint(self, checkpoint_path, model, optimizer=None, strict=True):
        """
        加载检查点
        Args:
            checkpoint_path: 检查点路径
            model: 模型
            optimizer: 优化器（可选）
            strict: 是否严格匹配参数
        Returns:
            start_epoch: 开始的epoch
        """
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return 0

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # 加载模型状态
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=strict)
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=strict)
        else:
            model.load_state_dict(checkpoint, strict=strict)

        # 加载优化器状态
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

        start_epoch = checkpoint.get('epoch', 0)

        print(f"Loaded checkpoint from epoch {start_epoch}")
        return start_epoch

    def save_stage_checkpoint(self, state, stage_name):
        """保存阶段性检查点（如pretraining, bridge_tuning完成时）"""
        if not self.is_main_process:
            return

        stage_path = self.checkpoint_dir / f'{stage_name}_final.pth'
        torch.save(state, stage_path)
        print(f"Saved {stage_name} checkpoint to {stage_path}")

    def check_early_stopping(self, current_metric):
        """
        检查是否应该早停
        Args:
            current_metric: 当前评价指标值
        Returns:
            should_stop: 是否应该停止训练
        """
        if self.early_stopping_patience is None:
            return False

        if self.best_metric is None:
            self.best_metric = current_metric
            return False

        # 检查是否有提升
        improvement = current_metric - self.best_metric
        if improvement > self.min_delta:
            self.best_metric = current_metric
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.early_stopping_patience:
                self.should_stop = True
                return True
            return False

    def reset_early_stopping(self):
        """重置早停状态（用于新的训练阶段）"""
        self.patience_counter = 0
        self.should_stop = False
        self.best_metric = None
