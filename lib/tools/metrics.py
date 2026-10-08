import numpy as np
import torch
import torch.nn.functional as F
import scipy.ndimage
from scipy.ndimage import label as scipy_label


class MetricsCalculator:
    """医学图像分割指标计算工具"""

    @staticmethod
    def dice_score(pred, target, num_classes=None, smooth=1e-5):
        """
        计算3D医学图像分割的Dice系数
        Args:
            pred: 预测结果 (B, C, H, W, D) 或 (B, H, W, D)
            target: 真实标签 (B, H, W, D) 或 (B, 1, H, W, D)
            num_classes: 类别数
            smooth: 平滑系数
        """
        # 处理预测结果
        if pred.dim() == 5:  # (B, C, H, W, D)
            pred = torch.argmax(pred, dim=1)

        # 处理目标标签
        if target.dim() == 5 and target.shape[1] == 1:  # (B, 1, H, W, D)
            target = target.squeeze(1)  # 移除通道维度

        if num_classes is None:
            num_classes = int(target.max().item()) + 1

        dice_scores = []
        for c in range(1, num_classes):  # 跳过背景
            pred_c = (pred == c).float()
            target_c = (target == c).float()

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            dice = (2.0 * intersection + smooth) / (union + smooth)
            dice_scores.append(dice.item())

        return np.mean(dice_scores) if dice_scores else 0.0

    @staticmethod
    def iou_score(pred, target, num_classes=None, smooth=1e-5):
        """计算3D医学图像分割的IoU分数"""
        # 处理预测结果
        if pred.dim() == 5:  # (B, C, H, W, D)
            pred = torch.argmax(pred, dim=1)

        # 处理目标标签
        if target.dim() == 5 and target.shape[1] == 1:  # (B, 1, H, W, D)
            target = target.squeeze(1)  # 移除通道维度

        if num_classes is None:
            num_classes = int(target.max().item()) + 1

        iou_scores = []
        for c in range(1, num_classes):
            pred_c = (pred == c).float()
            target_c = (target == c).float()

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum() - intersection

            iou = (intersection + smooth) / (union + smooth)
            iou_scores.append(iou.item())

        return np.mean(iou_scores) if iou_scores else 0.0

    @staticmethod
    def hausdorff_distance_95(pred, target, num_classes=None, voxel_spacing=(1.0, 1.0, 1.0)):
        """
        计算95% Hausdorff距离（HD95）
        Args:
            pred: 预测结果 (B, C, H, W, D) 或 (B, H, W, D)
            target: 真实标签 (B, H, W, D) 或 (B, 1, H, W, D)
            num_classes: 类别数
            voxel_spacing: 体素间距 (mm)
        """
        try:
            from scipy.ndimage import distance_transform_edt
        except ImportError:
            return 0.0

        # 处理预测结果
        if pred.dim() == 5:  # (B, C, H, W, D)
            pred = torch.argmax(pred, dim=1)

        # 处理目标标签
        if target.dim() == 5 and target.shape[1] == 1:  # (B, 1, H, W, D)
            target = target.squeeze(1)

        # 转换为numpy
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()

        if num_classes is None:
            num_classes = int(target.max()) + 1

        # 处理批次维度
        batch_size = pred.shape[0]
        all_hd95_scores = []

        for b in range(batch_size):
            pred_b = pred[b]
            target_b = target[b]

            hd95_scores = []
            for c in range(1, num_classes):  # 跳过背景
                pred_c = (pred_b == c).astype(np.uint8)
                target_c = (target_b == c).astype(np.uint8)

                # 如果预测或真实标签中没有该类别，跳过
                if pred_c.sum() == 0 and target_c.sum() == 0:
                    continue
                elif pred_c.sum() == 0 or target_c.sum() == 0:
                    # 如果只有一个为空，HD95设为一个较大的值
                    hd95_scores.append(100.0)
                    continue

                # 计算表面点
                # 通过边界检测获取表面点
                pred_border = pred_c - scipy.ndimage.binary_erosion(pred_c)
                target_border = target_c - scipy.ndimage.binary_erosion(target_c)

                # 如果没有边界点，跳过
                if pred_border.sum() == 0 or target_border.sum() == 0:
                    continue

                # 计算距离变换
                pred_distance = distance_transform_edt(1 - pred_border, sampling=voxel_spacing)
                target_distance = distance_transform_edt(1 - target_border, sampling=voxel_spacing)

                # 获取表面点的距离
                pred_surface_distances = target_distance[pred_border > 0]
                target_surface_distances = pred_distance[target_border > 0]

                # 合并所有表面距离
                all_surface_distances = np.concatenate([pred_surface_distances, target_surface_distances])

                # 计算95百分位数
                hd95 = np.percentile(all_surface_distances, 95)
                hd95_scores.append(hd95)

            if hd95_scores:
                all_hd95_scores.append(np.mean(hd95_scores))

        return np.mean(all_hd95_scores) if all_hd95_scores else 0.0

    @staticmethod
    def volume_metrics(pred, target, num_classes=None, voxel_spacing=(1.0, 1.0, 1.0)):
        """计算3D体积相关指标"""
        # 处理预测结果
        if pred.dim() == 5:  # (B, C, H, W, D)
            pred = torch.argmax(pred, dim=1)

        # 处理目标标签
        if target.dim() == 5 and target.shape[1] == 1:  # (B, 1, H, W, D)
            target = target.squeeze(1)  # 移除通道维度

        if num_classes is None:
            num_classes = int(target.max().item()) + 1

        metrics = {}
        voxel_volume = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]

        for c in range(1, num_classes):
            pred_c = (pred == c).float()
            target_c = (target == c).float()

            # 预测体积 (mm³)
            pred_volume = pred_c.sum().item() * voxel_volume
            # 真实体积 (mm³)
            target_volume = target_c.sum().item() * voxel_volume

            # 体积误差
            volume_error = abs(pred_volume - target_volume)
            relative_volume_error = volume_error / (target_volume + 1e-8)

            metrics[f'volume_error_class_{c}'] = volume_error
            metrics[f'relative_volume_error_class_{c}'] = relative_volume_error
            metrics[f'pred_volume_class_{c}'] = pred_volume
            metrics[f'target_volume_class_{c}'] = target_volume

        return metrics

    @staticmethod
    def compute_all_metrics(pred, target, num_classes=None, voxel_spacing=(1.0, 1.0, 1.0)):
        """计算所有3D医学图像分割指标"""
        metrics = {}
        metrics['dice'] = MetricsCalculator.dice_score(pred, target, num_classes)
        metrics['iou'] = MetricsCalculator.iou_score(pred, target, num_classes)
        metrics['hd95'] = MetricsCalculator.hausdorff_distance_95(pred, target, num_classes, voxel_spacing)

        # 添加体积指标
        volume_metrics = MetricsCalculator.volume_metrics(pred, target, num_classes, voxel_spacing)
        metrics.update(volume_metrics)

        return metrics
