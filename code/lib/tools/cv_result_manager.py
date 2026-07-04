"""
交叉验证结果管理工具

管理交叉验证结果的保存、验证和汇总
"""

import json
import numpy as np
from pathlib import Path


class CVResultManager:
    """管理交叉验证结果的保存和汇总"""

    def save_single_result(self, result, output_dir):
        """
        保存单次训练的结果

        Args:
            result: 结果字典
            output_dir: 输出目录路径
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / 'test_results.json', 'w') as f:
            json.dump(result, f, indent=2)

    def validate_result(self, result):
        """
        验证结果的有效性

        Args:
            result: 结果字典

        Raises:
            AssertionError: 如果结果无效
        """
        import numpy as np

        # 验证必需字段
        required_fields = ['peft_method', 'center', 'weight', 'fold',
                          'k_shot', 'seed', 'test_dice', 'test_iou', 'test_hd95']
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

        # 验证值不是NaN或Inf
        assert not np.isnan(result['test_dice']), f"Invalid Dice (NaN): {result['test_dice']}"
        assert not np.isinf(result['test_dice']), f"Invalid Dice (Inf): {result['test_dice']}"
        assert not np.isnan(result['test_iou']), f"Invalid IoU (NaN): {result['test_iou']}"
        assert not np.isinf(result['test_iou']), f"Invalid IoU (Inf): {result['test_iou']}"
        assert not np.isnan(result['test_hd95']), f"Invalid HD95 (NaN): {result['test_hd95']}"
        assert not np.isinf(result['test_hd95']), f"Invalid HD95 (Inf): {result['test_hd95']}"

        # 验证值范围
        assert 0 <= result['test_dice'] <= 1, f"Invalid Dice: {result['test_dice']}"
        assert 0 <= result['test_iou'] <= 1, f"Invalid IoU: {result['test_iou']}"
        assert result['test_hd95'] >= 0, f"Invalid HD95: {result['test_hd95']}"

        # 验证配置（放宽限制，避免把实验设计写死）
        assert int(result['k_shot']) > 0, f"Invalid k_shot: {result['k_shot']}"
        # seed允许任意整数（单seed/多seed都可）
        int(result['seed'])

        # 验证训练是否完成（至少训练了1个epoch）
        if 'epochs_trained' in result:
            assert result['epochs_trained'] > 0, f"Invalid epochs_trained: {result['epochs_trained']}"

    def compute_cv_summary(self, all_results, method_name, center_name,
                          weight_name, n_folds, k_shot_list, seed_list=None):
        """
        计算交叉验证汇总

        Args:
            all_results: 所有结果列表（包含所有折×shot×seed的结果）
            method_name: PEFT方法名称
            center_name: 中心名称
            weight_name: 权重名称
            n_folds: 折数
            k_shot_list: K-shot列表

        Returns:
            summary: 汇总字典
        """
        summary = {
            'peft_method': method_name,
            'center': center_name,
            'weight': weight_name,
            'n_folds': n_folds,
            'results': {}
        }

        for k_shot in k_shot_list:
            # 筛选当前shot的结果
            shot_results = [r for r in all_results if r['k_shot'] == k_shot]

            # 验证结果数量（默认兼容旧逻辑：每折3个seed）
            if seed_list is None:
                expected_count = n_folds * 3
            else:
                expected_count = n_folds * len(list(seed_list))
            if len(shot_results) != expected_count:
                print(f"Warning: Expected {expected_count} results for {k_shot}-shot, "
                      f"got {len(shot_results)}")

            # 按折分组，计算每折的平均值（3个seed的平均）
            fold_dices, fold_ious, fold_hd95s = [], [], []
            for fold_idx in range(n_folds):
                fold_seed_results = [r for r in shot_results if r['fold'] == fold_idx]

                if len(fold_seed_results) > 0:
                    fold_dices.append(np.mean([r['test_dice'] for r in fold_seed_results]))
                    fold_ious.append(np.mean([r['test_iou'] for r in fold_seed_results]))
                    fold_hd95s.append(np.mean([r['test_hd95'] for r in fold_seed_results]))
                else:
                    print(f"Warning: No results for fold {fold_idx}")

            # 计算所有折的统计（折间的平均值和标准差）
            summary['results'][f'{k_shot}shot'] = {
                'dice': {
                    'mean': float(np.mean(fold_dices)) if fold_dices else 0.0,
                    'std': float(np.std(fold_dices)) if fold_dices else 0.0,
                    'per_fold': [float(d) for d in fold_dices]
                },
                'iou': {
                    'mean': float(np.mean(fold_ious)) if fold_ious else 0.0,
                    'std': float(np.std(fold_ious)) if fold_ious else 0.0,
                    'per_fold': [float(d) for d in fold_ious]
                },
                'hd95': {
                    'mean': float(np.mean(fold_hd95s)) if fold_hd95s else 0.0,
                    'std': float(np.std(fold_hd95s)) if fold_hd95s else 0.0,
                    'per_fold': [float(d) for d in fold_hd95s]
                }
            }

        return summary

    def save_cv_summary(self, summary, output_path):
        """
        保存交叉验证汇总文件

        Args:
            summary: 汇总字典
            output_path: 输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"    Saved CV summary: {output_path}")

    def validate_cv_summary(self, summary, expected_folds):
        """
        验证交叉验证汇总的完整性

        Args:
            summary: 汇总字典
            expected_folds: 期望的折数

        Returns:
            bool: 是否通过验证
        """
        try:
            assert summary['n_folds'] == expected_folds
            assert 'results' in summary

            for shot_key in summary['results']:
                shot_data = summary['results'][shot_key]
                assert 'dice' in shot_data
                assert 'iou' in shot_data
                assert 'hd95' in shot_data
                assert len(shot_data['dice']['per_fold']) == expected_folds

            return True
        except AssertionError as e:
            print(f"CV summary validation failed: {e}")
            return False
