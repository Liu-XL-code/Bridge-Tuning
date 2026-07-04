import os
import csv
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from monai.data import list_data_collate

from .data_transforms import get_preload_transforms, get_augmentation_transforms


class MedicalSegmentationDataset(Dataset):
    """预加载到内存的医学图像分割数据集"""

    def __init__(self, data_path, list_file, sample_list, preload_transform, augmentation_transform=None, context=None):
        """
        Args:
            data_path: 数据根目录
            list_file: CSV数据列表文件
            sample_list: 样本列表（字典列表，优先级高于list_file）
            preload_transform: 预加载时的变换（一次性执行）
            augmentation_transform: 训练时的数据增强（每次执行）
        """
        self.augmentation_transform = augmentation_transform
        self.context = context

        # 优先使用sample_list，否则从CSV读取
        if sample_list is not None:
            file_list = self._process_sample_list(Path(data_path), sample_list)
        elif list_file is not None:
            file_list = self._load_csv_paths(Path(data_path), list_file)
        else:
            raise ValueError("Must provide either list_file or sample_list")

        ctx = f"[{self.context}] " if self.context else ""
        src = f"sample_list({len(sample_list)})" if sample_list is not None else f"csv({list_file})"
        # 用 tqdm.write 避免破坏进度条显示
        tqdm.write(f"{ctx}Loading {len(file_list)} samples to memory... (data_path={data_path}, source={src})")
        self.data_list = []
        tqdm_desc = f"Preloading {self.context}" if self.context else "Preloading data"
        pbar = tqdm(file_list, desc=tqdm_desc, dynamic_ncols=True, leave=True)
        for item in pbar:
            preprocessed = preload_transform(item)
            self.data_list.append(preprocessed)
        tqdm.write(f"{ctx}All {len(self.data_list)} samples loaded to memory")

    def _clean_path(self, path_str):
        """清洗路径字符串，去除两端空白及常见的标点符号"""
        if path_str is None:
            return path_str
        return str(path_str).strip().rstrip('、，,;；')

    def _process_sample_list(self, data_path, sample_list):
        """处理样本列表（字典列表）"""
        data_list = []
        for row in sample_list:
            item = {}
            for key in ['image', 'label']:
                if key in row:
                    path = self._clean_path(row[key])
                    item[key] = path if os.path.isabs(path) else str(data_path / path)
            data_list.append(item)
        return data_list

    def _load_csv_paths(self, data_path, list_file):
        """加载CSV数据路径列表"""
        data_list = []
        with open(list_file, 'r') as f:
            for row in csv.DictReader(f):
                item = {}
                for key in ['image', 'label']:
                    if key in row:
                        path = self._clean_path(row[key])
                        item[key] = path if os.path.isabs(path) else str(data_path / path)
                data_list.append(item)
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_dict = {
            'image': self.data_list[idx]['image'].clone(),
            'label': self.data_list[idx]['label'].clone()
        }

        if self.augmentation_transform:
            data_dict = self.augmentation_transform(data_dict)

        return data_dict


def get_dataloader(data_path, list_file, sample_list, batch_size, num_workers,
                   is_train=True, distributed=False, args=None, use_preload=True, context=None):
    """
    创建数据加载器（预加载模式）

    Args:
        data_path: 数据根目录
        list_file: CSV数据列表文件
        batch_size: batch大小
        num_workers: 工作线程数（预加载模式下自动设为0）
        is_train: 是否为训练模式
        distributed: 是否使用分布式训练
        args: 其他参数
        use_preload: 兼容参数（始终使用预加载）

    Returns:
        DataLoader
    """
    # 创建数据集
    dataset = MedicalSegmentationDataset(
        data_path=data_path,
        list_file=list_file,
        sample_list=sample_list,  # 新增参数
        preload_transform=get_preload_transforms(args, is_train=is_train),
        augmentation_transform=get_augmentation_transforms(args) if is_train else None,
        context=context
    )

    # 创建sampler
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True) \
              if distributed and is_train else None

    # 创建dataloader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and is_train),
        num_workers=0,  # 预加载模式不需要多线程
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=list_data_collate,
    )
