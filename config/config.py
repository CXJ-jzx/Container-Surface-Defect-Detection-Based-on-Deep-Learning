#全局配置
# config/config.py
"""
全局配置文件
"""

import os
from pathlib import Path
import platform
import torch
import matplotlib.pyplot as plt


class Config:
    """项目配置类"""

    # ==================== 路径配置 ====================
    # 项目根目录
    PROJECT_ROOT = Path(__file__).parent.parent.absolute()

    # 数据集路径
    DATASET_PATH = PROJECT_ROOT / 'dataset'

    # 输出路径
    OUTPUT_DIR = PROJECT_ROOT / 'outputs'
    MODEL_DIR = OUTPUT_DIR / 'models'
    LOG_DIR = OUTPUT_DIR / 'logs'
    RESULT_DIR = OUTPUT_DIR / 'results'
    FIGURE_DIR = OUTPUT_DIR / 'figures'

    # 配置文件路径
    CONFIG_DIR = PROJECT_ROOT / 'config'
    DATASET_YAML = CONFIG_DIR / 'container_dataset.yaml'

    # ==================== 数据集配置 ====================
    # 类别信息
    CLASSES = ['dent', 'hole', 'rusty']
    CLASSES_CN = ['凹陷', '破洞', '锈蚀']
    NUM_CLASSES = len(CLASSES)

    # 类别颜色 (RGB)
    CLASS_COLORS = {
        0: (255, 0, 0),  # 凹陷 - 红色
        1: (0, 255, 0),  # 破洞 - 绿色
        2: (0, 0, 255)  # 锈蚀 - 蓝色
    }

    # ==================== 模型配置 ====================
    # 分类模型
    CLASSIFIER_BACKBONE = 'resnet50'
    CLASSIFIER_PRETRAINED = True
    CLASSIFIER_WEIGHTS = MODEL_DIR / 'best_classifier.pth'

    # 检测模型
    DETECTOR_MODEL_SIZE = 'm'  # n, s, m, l, x
    DETECTOR_WEIGHTS = MODEL_DIR / 'best_detector.pt'

    # ==================== 训练配置 ====================
    # 通用
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED = 42
    NUM_WORKERS = 4

    # 分类模型训练
    CLS_EPOCHS = 30
    CLS_BATCH_SIZE = 32
    CLS_LEARNING_RATE = 1e-4
    CLS_WEIGHT_DECAY = 1e-4
    CLS_IMAGE_SIZE = 224

    # 检测模型训练
    DET_EPOCHS = 100
    DET_BATCH_SIZE = 16
    DET_IMAGE_SIZE = 640
    DET_PATIENCE = 50

    # ==================== 推理配置 ====================
    CONFIDENCE_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45

    # ==================== 可视化配置 ====================
    FIGURE_DPI = 150
    FIGURE_FORMAT = 'png'

    @classmethod
    def setup(cls):
        """初始化配置，创建必要的目录"""
        # 创建输出目录
        dirs_to_create = [
            cls.OUTPUT_DIR,
            cls.MODEL_DIR,
            cls.LOG_DIR,
            cls.RESULT_DIR,
            cls.FIGURE_DIR
        ]

        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)

        # 设置中文字体
        cls._setup_chinese_font()

        # 设置随机种子
        cls._set_seed(cls.SEED)

        print(f"项目根目录: {cls.PROJECT_ROOT}")
        print(f"数据集路径: {cls.DATASET_PATH}")
        print(f"使用设备: {cls.DEVICE}")

    @classmethod
    def _setup_chinese_font(cls):
        """配置中文字体"""
        system = platform.system()

        if system == 'Windows':
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
        elif system == 'Darwin':
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
        else:
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']

        plt.rcParams['axes.unicode_minus'] = False

    @classmethod
    def _set_seed(cls, seed):
        """设置随机种子"""
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

    @classmethod
    def get_class_name(cls, class_id, chinese=False):
        """获取类别名称"""
        if chinese:
            return cls.CLASSES_CN[class_id]
        return cls.CLASSES[class_id]

    @classmethod
    def get_class_color(cls, class_id):
        """获取类别颜色"""
        return cls.CLASS_COLORS.get(class_id, (128, 128, 128))

    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("\n" + "=" * 60)
        print("项目配置")
        print("=" * 60)
        print(f"项目根目录: {cls.PROJECT_ROOT}")
        print(f"数据集路径: {cls.DATASET_PATH}")
        print(f"设备: {cls.DEVICE}")
        print(f"类别: {cls.CLASSES}")
        print(f"分类模型: {cls.CLASSIFIER_BACKBONE}")
        print(f"检测模型: YOLOv8{cls.DETECTOR_MODEL_SIZE}")
        print("=" * 60 + "\n")


def get_config():
    """获取配置实例"""
    Config.setup()
    return Config