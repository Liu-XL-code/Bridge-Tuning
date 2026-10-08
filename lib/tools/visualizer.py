import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib


class Visualizer:
    """训练过程可视化工具"""

    def __init__(self, output_dir=None, rank=0):
        if output_dir is not None:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = None
        self.rank = rank
        self.is_main_process = (rank == 0)

        if self.is_main_process and self.output_dir is not None:
            self.vis_dir = self.output_dir / 'visualizations'
            self.vis_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.vis_dir = None

    def set_output_dir(self, output_dir):
        """设置输出目录"""
        self.output_dir = Path(output_dir)
        if self.is_main_process:
            self.vis_dir = self.output_dir / 'visualizations'
            self.vis_dir.mkdir(parents=True, exist_ok=True)

    def save_training_curves(self, history, stage_name='training'):
        """保存训练曲线"""
        if not self.is_main_process:
            return

        # 绘制loss曲线
        if 'loss' in history:
            plt.figure(figsize=(10, 6))
            plt.plot(history['epoch'], history['loss'], label='Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'{stage_name} - Loss Curve')
            plt.legend()
            plt.grid(True)
            plt.savefig(self.vis_dir / f'{stage_name}_loss_curve.png', dpi=150, bbox_inches='tight')
            plt.close()

        # 绘制指标曲线
        metric_keys = [k for k in history.keys() if k not in ['epoch', 'loss', 'lr']]
        if metric_keys:
            plt.figure(figsize=(10, 6))
            for key in metric_keys:
                plt.plot(history['epoch'], history[key], label=key)
            plt.xlabel('Epoch')
            plt.ylabel('Metrics')
            plt.title(f'{stage_name} - Metrics Curve')
            plt.legend()
            plt.grid(True)
            plt.savefig(self.vis_dir / f'{stage_name}_metrics_curve.png', dpi=150, bbox_inches='tight')
            plt.close()

    def save_prediction_samples(self, images, labels, predictions, epoch, num_samples=4):
        """保存3D医学图像预测样本的可视化（已禁用）"""
        # Sample visualization is disabled in this training callback.
        pass

    def save_3d_volume(self, volume, filename, affine=None):
        """保存3D体积为NIfTI格式"""
        if not self.is_main_process:
            return

        if torch.is_tensor(volume):
            volume = volume.cpu().numpy()

        # Use uint8 for integer segmentation labels and float32 for floating-point volumes.
        if volume.dtype in [np.int64, np.int32]:
            volume = volume.astype(np.uint8)
        elif volume.dtype == np.float64:
            volume = volume.astype(np.float32)
        elif volume.dtype not in [np.uint8, np.int16, np.int32, np.float32]:
            # 对于其他不兼容的类型，转换为float32
            volume = volume.astype(np.float32)

        if affine is None:
            affine = np.eye(4)

        # Create the NIfTI image after converting the volume to a supported storage dtype.
        nii_img = nib.Nifti1Image(volume, affine)
        nib.save(nii_img, self.vis_dir / filename)

    def save_feature_maps(self, features, epoch, layer_name='features'):
        """保存3D特征图可视化"""
        if not self.is_main_process:
            return

        if torch.is_tensor(features):
            features = features.cpu().numpy()

        # 选择前16个通道进行可视化
        num_channels = min(16, features.shape[1])

        # 为每个通道创建3个正交切面的可视化
        for i in range(num_channels):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # 获取3D特征图
            if features.ndim == 5:  # (B, C, H, W, D)
                feat_volume = features[0, i]  # 移除batch和通道维度
            else:
                feat_volume = features[0, i]

            # 获取中间切片索引
            mid_h = feat_volume.shape[0] // 2
            mid_w = feat_volume.shape[1] // 2
            mid_d = feat_volume.shape[2] // 2

            # 轴向切片
            axes[0].imshow(feat_volume[mid_h, :, :], cmap='viridis')
            axes[0].set_title(f'Channel {i} - Axial')
            axes[0].axis('off')

            # 冠状切片
            axes[1].imshow(feat_volume[:, mid_w, :], cmap='viridis')
            axes[1].set_title(f'Channel {i} - Coronal')
            axes[1].axis('off')

            # 矢状切片
            axes[2].imshow(feat_volume[:, :, mid_d], cmap='viridis')
            axes[2].set_title(f'Channel {i} - Sagittal')
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig(self.vis_dir / f'{layer_name}_epoch_{epoch:04d}_channel_{i:02d}.png',
                       dpi=150, bbox_inches='tight')
            plt.close()

    def plot_training_curves(self, history, dataset_name, k_shot, sampling_seed, logger=None):
        """
        绘制训练过程曲线

        Args:
            history: 训练历史字典
            dataset_name: 数据集名称
            k_shot: K-shot值
            sampling_seed: 采样种子
            logger: 日志记录器（可选）
        """
        if not self.is_main_process or self.vis_dir is None:
            return

        if not history or 'epoch' not in history:
            if logger:
                logger.warning("No training history to plot")
            return

        epochs = history['epoch']

        # 设置字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 图1: Loss曲线
        if 'loss' in history:
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, history['loss'], 'b-', linewidth=2, label='Loss')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title(f'{dataset_name} - {k_shot}shot - seed{sampling_seed} - Loss Curve', fontsize=14)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            loss_file = self.vis_dir / 'loss_curve.png'
            plt.savefig(loss_file, dpi=150, bbox_inches='tight')
            plt.close()
            if logger:
                logger.info(f"Loss curve saved to: {loss_file}")

        # 图2: Dice和IoU曲线（放在一起，因为数量级相近）
        if 'dice' in history and 'iou' in history:
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, history['dice'], 'r-', linewidth=2, label='Dice', marker='o', markersize=3, markevery=max(1, len(epochs)//20))
            plt.plot(epochs, history['iou'], 'g-', linewidth=2, label='IoU', marker='s', markersize=3, markevery=max(1, len(epochs)//20))
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Score', fontsize=12)
            plt.title(f'{dataset_name} - {k_shot}shot - seed{sampling_seed} - Dice & IoU Curves', fontsize=14)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.ylim([0, 1])
            metrics_file = self.vis_dir / 'dice_iou_curves.png'
            plt.savefig(metrics_file, dpi=150, bbox_inches='tight')
            plt.close()
            if logger:
                logger.info(f"Dice & IoU curves saved to: {metrics_file}")

        # 图3: HD95曲线（单独，因为数量级不同）
        if 'hd95' in history:
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, history['hd95'], 'm-', linewidth=2, label='HD95', marker='^', markersize=3, markevery=max(1, len(epochs)//20))
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('HD95 (mm)', fontsize=12)
            plt.title(f'{dataset_name} - {k_shot}shot - seed{sampling_seed} - HD95 Curve', fontsize=14)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            hd95_file = self.vis_dir / 'hd95_curve.png'
            plt.savefig(hd95_file, dpi=150, bbox_inches='tight')
            plt.close()
            if logger:
                logger.info(f"HD95 curve saved to: {hd95_file}")

        if logger:
            logger.info("All training curves saved successfully")
