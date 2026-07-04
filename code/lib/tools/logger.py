import os
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import torch.distributed as dist


class TrainingLogger:
    """训练过程记录工具"""

    def __init__(self, log_dir, rank=0, distributed=False):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.distributed = distributed
        self.is_main_process = (rank == 0)

        # 训练历史记录
        self.history = defaultdict(list)

        # 最优指标记录
        self.best_metrics = {
            'epoch': -1,
            'dice': 0.0,
            'iou': 0.0,
            'hd95': float('inf')
        }

        # 设置logger
        self._setup_logger()

    def _setup_logger(self):
        """设置日志记录器"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f'train_{timestamp}.log'

        # 使用唯一的logger名称
        logger_name = f"Trainer_{self.rank}_{timestamp}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)

        # 清除所有旧的handlers
        self.logger.handlers.clear()

        # 文件handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)

        # 控制台handler
        if self.is_main_process:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(message)s')
            )
            self.logger.addHandler(stream_handler)

    def log_config(self, config):
        """记录配置信息"""
        if not self.is_main_process:
            return

        self.logger.info("=" * 60)
        self.logger.info("Training Configuration")
        self.logger.info("=" * 60)

        config_dict = vars(config) if hasattr(config, '__dict__') else config
        for key, value in config_dict.items():
            self.logger.info(f"{key:30s}: {value}")
        self.logger.info("=" * 60)

        # 保存配置到JSON
        config_file = self.log_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

    def log_epoch(self, epoch, metrics_dict):
        """记录每个epoch的指标"""
        if not self.is_main_process:
            return

        # 更新历史
        self.history['epoch'].append(epoch)
        for key, value in metrics_dict.items():
            self.history[key].append(value)

        # 打印日志
        msg = f"Epoch [{epoch}] "
        msg += " | ".join([f"{k}: {v:.4f}" for k, v in metrics_dict.items()])
        self.logger.info(msg)

        # 保存历史到文件
        history_file = self.log_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(dict(self.history), f, indent=2)

    def log_step(self, step, metrics_dict, print_freq=10):
        """记录训练步骤"""
        if not self.is_main_process:
            return

        if step % print_freq == 0:
            msg = f"Step [{step}] "
            msg += " | ".join([f"{k}: {v:.4f}" for k, v in metrics_dict.items()])
            self.logger.info(msg)

    def info(self, msg):
        """记录信息"""
        if self.is_main_process:
            self.logger.info(msg)

    def warning(self, msg):
        """记录警告"""
        if self.is_main_process:
            self.logger.warning(msg)

    def error(self, msg):
        """记录错误"""
        if self.is_main_process:
            self.logger.error(msg)

    def debug(self, msg):
        """记录调试信息"""
        if self.is_main_process:
            self.logger.debug(msg)

    def update_best_metrics(self, epoch, metrics_dict):
        """更新最优指标"""
        if not self.is_main_process:
            return

        dice = metrics_dict.get('val_dice', metrics_dict.get('dice', 0.0))

        # 如果当前dice更好，更新所有最优指标
        if dice > self.best_metrics['dice']:
            self.best_metrics['epoch'] = epoch
            self.best_metrics['dice'] = dice
            self.best_metrics['iou'] = metrics_dict.get('val_iou', metrics_dict.get('iou', 0.0))
            self.best_metrics['hd95'] = metrics_dict.get('val_hd95', metrics_dict.get('hd95', 0.0))

            # 保存最优指标到文件
            best_metrics_file = self.log_dir / 'best_metrics.json'
            with open(best_metrics_file, 'w') as f:
                json.dump(self.best_metrics, f, indent=2)

            self.logger.info(f">>> New best metrics at epoch {epoch}: Dice={dice:.4f}, IoU={self.best_metrics['iou']:.4f}, HD95={self.best_metrics['hd95']:.4f}")

    def save_training_curve(self):
        """保存训练曲线数据"""
        if not self.is_main_process:
            return

        curve_file = self.log_dir / 'training_curve.json'
        with open(curve_file, 'w') as f:
            json.dump(dict(self.history), f, indent=2)

    def reset(self):
        """重置logger状态（用于新的训练阶段）"""
        self.history = defaultdict(list)
        self.best_metrics = {
            'epoch': -1,
            'dice': 0.0,
            'iou': 0.0,
            'hd95': float('inf')
        }
