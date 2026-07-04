"""
交叉验证数据划分工具

提供可复现的K折交叉验证划分功能
"""

import random
import numpy as np


def split_kfold_cv(samples, n_folds, cv_seed):
    """
    K折交叉验证划分（可复现）

    Args:
        samples: 样本列表（CSV行字典列表）
        n_folds: 折数
        cv_seed: 随机种子（用于可复现划分）

    Returns:
        List[(train_samples, test_samples)]:
            每个元素是一个折的训练集和测试集样本列表

    示例:
        >>> samples = [{'image': 'img1.nii.gz', 'label': 'lab1.nii.gz'}, ...]
        >>> folds = split_kfold_cv(samples, n_folds=4, cv_seed=42)
        >>> train, test = folds[0]
    """
    # 设置随机种子以确保可复现
    random.seed(cv_seed)
    np.random.seed(cv_seed)

    # 打乱样本
    shuffled = random.sample(samples, len(samples))

    # 计算每折大小
    fold_size = len(shuffled) // n_folds

    folds = []
    for i in range(n_folds):
        # 当前折作为测试集
        if i < n_folds - 1:
            test_samples = shuffled[i * fold_size : (i+1) * fold_size]
        else:
            # 最后一折包含剩余所有样本
            test_samples = shuffled[i * fold_size :]

        # 其余样本作为训练集
        train_samples = [s for s in shuffled if s not in test_samples]

        folds.append((train_samples, test_samples))

    return folds


def generate_cv_seed(center_name):
    """
    生成可复现的交叉验证种子

    Args:
        center_name: center name, for example 'TARGET_A'

    Returns:
        int: 可复现的随机种子

    注意：
        - 相同中心名生成相同种子
        - 确保交叉验证划分的可复现性
    """
    return hash(center_name) % (2**31)


def sample_kshot(samples, k_shot, seed):
    """
    从样本中采样K个样本（可复现）

    Args:
        samples: 样本列表
        k_shot: 采样数量
        seed: 随机种子

    Returns:
        List: K个采样的样本
    """
    random.seed(seed)
    return random.sample(samples, min(k_shot, len(samples)))
