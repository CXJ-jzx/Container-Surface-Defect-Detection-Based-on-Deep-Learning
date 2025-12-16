# data/dataset.py
"""
数据集定义
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from pathlib import Path
import numpy as np

import sys

sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from data.transforms import get_train_transforms, get_test_transforms


class ClassificationDataset(Dataset):
    """
    多标签分类数据集
    用于判断图片中存在哪些类型的缺陷
    """

    def __init__(self, image_dir, label_dir, transform=None, num_classes=3):
        """
        Args:
            image_dir: 图片目录路径
            label_dir: 标签目录路径
            transform: 数据变换
            num_classes: 类别数量
        """
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.num_classes = num_classes

        # 获取所有图片文件
        self.images = self._get_image_files()

        if len(self.images) == 0:
            raise ValueError(f"未在 {image_dir} 中找到图片文件")

        print(f"加载数据集: {len(self.images)} 张图片")

    def _get_image_files(self):
        """获取所有图片文件（修复重复问题）"""
        # 使用集合去重，统一转换为小写比较
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

        images = []
        seen_files = set()  # 用于去重

        for file_path in self.image_dir.iterdir():
            if file_path.is_file():
                # 检查扩展名（不区分大小写）
                ext = file_path.suffix.lower()
                if ext in image_extensions:
                    # 使用小写文件名作为唯一标识（避免Windows不区分大小写的问题）
                    file_key = str(file_path).lower()
                    if file_key not in seen_files:
                        seen_files.add(file_key)
                        images.append(file_path)

        return sorted(images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 获取图片路径
        img_path = self.images[idx]
        label_path = self.label_dir / (img_path.stem + '.txt')

        # 读取图片
        image = Image.open(img_path).convert('RGB')

        # 读取标签 - 多标签格式
        labels = torch.zeros(self.num_classes, dtype=torch.float32)

        if label_path.exists():
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        try:
                            class_id = int(parts[0])
                            if 0 <= class_id < self.num_classes:
                                labels[class_id] = 1.0
                        except ValueError:
                            continue

        # 应用数据变换
        if self.transform:
            image = self.transform(image)

        return image, labels, str(img_path)


class DetectionDataset(Dataset):
    """
    目标检测数据集
    用于YOLO格式的数据加载（主要用于数据分析，训练使用ultralytics内置的）
    """

    def __init__(self, image_dir, label_dir, transform=None):
        """
        Args:
            image_dir: 图片目录路径
            label_dir: 标签目录路径
            transform: 数据变换
        """
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform

        # 获取所有图片文件
        self.images = self._get_image_files()

        print(f"检测数据集: {len(self.images)} 张图片")

    def _get_image_files(self):
        """获取所有图片文件（修复重复问题）"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

        images = []
        seen_files = set()

        for file_path in self.image_dir.iterdir():
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in image_extensions:
                    file_key = str(file_path).lower()
                    if file_key not in seen_files:
                        seen_files.add(file_key)
                        images.append(file_path)

        return sorted(images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.label_dir / (img_path.stem + '.txt')

        # 读取图片
        image = Image.open(img_path).convert('RGB')
        img_width, img_height = image.size

        # 读取标签 - YOLO格式
        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])

                            # 转换为像素坐标
                            x1 = (x_center - width / 2) * img_width
                            y1 = (y_center - height / 2) * img_height
                            x2 = (x_center + width / 2) * img_width
                            y2 = (y_center + height / 2) * img_height

                            boxes.append([x1, y1, x2, y2])
                            labels.append(class_id)
                        except ValueError:
                            continue

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, boxes, labels, str(img_path)


def get_data_loaders(dataset_path, batch_size=32, image_size=224,
                     val_split=0.1, num_workers=4):
    """
    获取数据加载器

    Args:
        dataset_path: 数据集根目录
        batch_size: 批次大小
        image_size: 图像大小
        val_split: 验证集比例
        num_workers: 数据加载线程数

    Returns:
        train_loader, val_loader, test_loader
    """
    dataset_path = Path(dataset_path)

    # 获取数据变换
    train_transform = get_train_transforms(image_size)
    test_transform = get_test_transforms(image_size)

    # 创建训练集
    train_dataset = ClassificationDataset(
        image_dir=dataset_path / 'images' / 'train',
        label_dir=dataset_path / 'labels' / 'train',
        transform=train_transform
    )

    # 创建测试集
    test_dataset = ClassificationDataset(
        image_dir=dataset_path / 'images' / 'test',
        label_dir=dataset_path / 'labels' / 'test',
        transform=test_transform
    )

    # 划分验证集
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\n数据加载器创建完成:")
    print(f"  训练集: {len(train_subset)} 样本")
    print(f"  验证集: {len(val_subset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")

    return train_loader, val_loader, test_loader


def verify_dataset(dataset_path):
    """
    验证数据集，检查是否有重复文件

    Args:
        dataset_path: 数据集路径
    """
    dataset_path = Path(dataset_path)

    print("\n" + "=" * 50)
    print("数据集验证")
    print("=" * 50)

    for split in ['train', 'val', 'test']:
        image_dir = dataset_path / 'images' / split
        label_dir = dataset_path / 'labels' / split

        if not image_dir.exists():
            continue

        # 统计图片
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = set()
        duplicates = []

        for file_path in image_dir.iterdir():
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in image_extensions:
                    file_key = file_path.stem.lower()
                    if file_key in images:
                        duplicates.append(file_path.name)
                    else:
                        images.add(file_key)

        # 统计标签
        labels = list(label_dir.glob('*.txt')) if label_dir.exists() else []

        print(f"\n{split.upper()}集:")
        print(f"  图片数量: {len(images)}")
        print(f"  标签数量: {len(labels)}")

        if duplicates:
            print(f"  ⚠️ 发现重复文件: {len(duplicates)}")
            for dup in duplicates[:5]:
                print(f"    - {dup}")
            if len(duplicates) > 5:
                print(f"    ... 还有 {len(duplicates) - 5} 个")
        else:
            print(f"  ✓ 无重复文件")

        # 检查图片和标签是否匹配
        image_stems = {img.lower() for img in images}
        label_stems = {Path(lbl).stem.lower() for lbl in labels}

        missing_labels = image_stems - label_stems
        extra_labels = label_stems - image_stems

        if missing_labels:
            print(f"  ⚠️ 缺少标签的图片: {len(missing_labels)}")
        if extra_labels:
            print(f"  ⚠️ 多余的标签文件: {len(extra_labels)}")
        if not missing_labels and not extra_labels:
            print(f"  ✓ 图片和标签完全匹配")

    print("\n" + "=" * 50)


if __name__ == '__main__':
    # 测试数据集验证
    verify_dataset('./dataset')