# utils/helpers.py
"""
辅助函数
"""

import os
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import json
import yaml


def set_seed(seed=42):
    """
    设置随机种子

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"随机种子已设置: {seed}")


def get_timestamp():
    """获取当前时间戳字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_directory(path, parents=True, exist_ok=True):
    """
    创建目录

    Args:
        path: 目录路径
        parents: 是否创建父目录
        exist_ok: 是否允许目录已存在

    Returns:
        Path对象
    """
    path = Path(path)
    path.mkdir(parents=parents, exist_ok=exist_ok)
    return path


def save_json(data, filepath):
    """
    保存JSON文件

    Args:
        data: 要保存的数据
        filepath: 文件路径
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"JSON已保存: {filepath}")


def load_json(filepath):
    """
    加载JSON文件

    Args:
        filepath: 文件路径

    Returns:
        加载的数据
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_yaml(data, filepath):
    """
    保存YAML文件

    Args:
        data: 要保存的数据
        filepath: 文件路径
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    print(f"YAML已保存: {filepath}")


def load_yaml(filepath):
    """
    加载YAML文件

    Args:
        filepath: 文件路径

    Returns:
        加载的数据
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_device():
    """获取可用的计算设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    return device


def count_parameters(model):
    """
    统计模型参数量

    Args:
        model: PyTorch模型

    Returns:
        参数量字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params,
        'total_mb': total_params * 4 / (1024 ** 2),  # 假设float32
    }


def format_time(seconds):
    """
    格式化时间

    Args:
        seconds: 秒数

    Returns:
        格式化的时间字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


class AverageMeter:
    """计算和存储平均值"""

    def __init__(self, name='Value'):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=10, min_delta=0, mode='max'):
        """
        Args:
            patience: 容忍次数
            min_delta: 最小改善值
            mode: 'max' 或 'min'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


def get_file_size(filepath):
    """获取文件大小(MB)"""
    size_bytes = os.path.getsize(filepath)
    return size_bytes / (1024 * 1024)