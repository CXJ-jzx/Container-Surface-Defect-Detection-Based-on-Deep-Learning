# models/detector.py
"""
YOLO目标检测模型
用于定位缺陷位置并识别类别
"""

import os
import yaml
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

import sys

sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config

try:
    from ultralytics import YOLO
except ImportError:
    print("请安装ultralytics: pip install ultralytics")


def create_dataset_yaml(dataset_path=None, output_path=None):
    """
    创建YOLO格式的数据集配置文件

    Args:
        dataset_path: 数据集路径
        output_path: 输出路径

    Returns:
        配置文件路径
    """
    dataset_path = Path(dataset_path) if dataset_path else Config.DATASET_PATH
    output_path = Path(output_path) if output_path else Config.CONFIG_DIR

    dataset_path = dataset_path.absolute()
    output_path.mkdir(parents=True, exist_ok=True)

    # 检查验证集是否存在
    val_path = 'images/val' if (dataset_path / 'images/val').exists() else 'images/test'

    config = {
        'path': str(dataset_path),
        'train': 'images/train',
        'val': val_path,
        'test': 'images/test',
        'nc': Config.NUM_CLASSES,
        'names': {i: name for i, name in enumerate(Config.CLASSES)}
    }

    yaml_path = output_path / 'container_dataset.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"数据集配置已创建: {yaml_path}")

    return str(yaml_path)


class ContainerDetector:
    """集装箱缺陷检测器"""

    def __init__(self, model_size='m', device=None):
        """
        Args:
            model_size: 模型大小 ('n', 's', 'm', 'l', 'x')
            device: 计算设备
        """
        self.model_size = model_size
        self.device = device or Config.DEVICE
        self.model = None

        self.classes = Config.CLASSES
        self.classes_cn = Config.CLASSES_CN
        self.colors = Config.CLASS_COLORS

        print(f"检测器初始化")
        print(f"  模型大小: YOLOv8{model_size}")
        print(f"  设备: {self.device}")

    def load_model(self, weights_path):
        """加载模型权重"""
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")

        self.model = YOLO(str(weights_path))
        print(f"模型已加载: {weights_path}")

        return self.model

    def train(self, data_yaml, epochs=100, imgsz=640, batch=16,
              project=None, name=None, resume=False):
        """
        训练检测模型

        Args:
            data_yaml: 数据集配置文件路径
            epochs: 训练轮数
            imgsz: 输入图像大小
            batch: 批次大小
            project: 项目保存目录
            name: 实验名称
            resume: 是否恢复训练

        Returns:
            训练结果
        """
        print("\n" + "=" * 60)
        print("开始训练YOLO检测模型")
        print("=" * 60)

        # 设置默认保存路径
        if project is None:
            project = str(Config.OUTPUT_DIR / 'detector_runs')
        if name is None:
            name = f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

        # 加载预训练模型
        model_name = f'yolov8{self.model_size}.pt'
        print(f"加载预训练模型: {model_name}")
        self.model = YOLO(model_name)

        # 训练参数
        train_args = {
            'data': data_yaml,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'project': project,
            'name': name,
            'device': 0 if self.device == 'cuda' else 'cpu',
            'workers': Config.NUM_WORKERS,
            'patience': Config.DET_PATIENCE,
            'save': True,
            'save_period': 10,
            'exist_ok': True,
            'resume': resume,

            # 优化器参数
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,

            # 数据增强
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 10.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 2.0,
            'perspective': 0.0001,
            'flipud': 0.5,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.1,

            # 其他
            'plots': True,
            'val': True,
            'verbose': True,
        }

        print(f"\n训练配置:")
        print(f"  数据集: {data_yaml}")
        print(f"  轮数: {epochs}")
        print(f"  批次: {batch}")
        print(f"  图像大小: {imgsz}")

        # 开始训练
        results = self.model.train(**train_args)

        # 保存最佳模型到指定目录
        best_pt = Path(project) / name / 'weights' / 'best.pt'
        if best_pt.exists():
            target_path = Config.MODEL_DIR / 'best_detector.pt'
            import shutil
            shutil.copy(best_pt, target_path)
            print(f"\n最佳模型已复制到: {target_path}")

        print("\n训练完成!")

        return results

    def validate(self, data_yaml, weights_path=None, split='test', conf=0.25):
        """
        验证模型

        Args:
            data_yaml: 数据集配置文件
            weights_path: 模型权重路径
            split: 验证集划分
            conf: 置信度阈值

        Returns:
            验证结果
        """
        print("\n" + "=" * 60)
        print("验证检测模型")
        print("=" * 60)

        if weights_path:
            self.load_model(weights_path)

        if self.model is None:
            raise ValueError("请先加载模型")

        results = self.model.val(
            data=data_yaml,
            split=split,
            conf=conf,
            plots=True,
            verbose=True
        )

        # 打印结果
        print("\n验证结果:")
        print("-" * 40)
        print(f"mAP@0.5:      {results.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {results.box.map:.4f}")
        print(f"Precision:    {results.box.mp:.4f}")
        print(f"Recall:       {results.box.mr:.4f}")

        print("\n各类别 AP@0.5:")
        for i, (name, name_cn) in enumerate(zip(self.classes, self.classes_cn)):
            if i < len(results.box.ap50):
                ap = results.box.ap50[i]
                print(f"  {name_cn} ({name}): {ap:.4f}")

        return results

    def predict_single(self, image_path, conf=0.25, iou=0.45):
        """
        预测单张图片

        Args:
            image_path: 图片路径
            conf: 置信度阈值
            iou: NMS IoU阈值

        Returns:
            预测结果字典
        """
        if self.model is None:
            raise ValueError("请先加载模型")

        results = self.model.predict(
            source=str(image_path),
            conf=conf,
            iou=iou,
            verbose=False
        )

        result = results[0]

        # 解析检测结果
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            detections.append({
                'bbox': {
                    'x1': int(x1), 'y1': int(y1),
                    'x2': int(x2), 'y2': int(y2),
                    'width': int(x2 - x1),
                    'height': int(y2 - y1)
                },
                'confidence': confidence,
                'class_id': class_id,
                'class_name': self.classes[class_id],
                'class_name_cn': self.classes_cn[class_id]
            })

        # 统计
        class_counts = {name: 0 for name in self.classes_cn}
        for det in detections:
            class_counts[det['class_name_cn']] += 1

        return {
            'image_path': str(image_path),
            'has_defect': len(detections) > 0,
            'total_defects': len(detections),
            'class_counts': class_counts,
            'detections': detections
        }

    def predict_batch(self, image_dir, output_dir=None, conf=0.25,
                      save_images=True, save_json=True):
        """
        批量预测

        Args:
            image_dir: 图片目录
            output_dir: 输出目录
            conf: 置信度阈值
            save_images: 是否保存可视化图片
            save_json: 是否保存JSON结果

        Returns:
            所有预测结果和统计
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir) if output_dir else Config.RESULT_DIR / 'predictions'
        output_dir.mkdir(parents=True, exist_ok=True)

        # 获取图片
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in extensions:
            image_files.extend(list(image_dir.glob(ext)))
            image_files.extend(list(image_dir.glob(ext.upper())))

        if not image_files:
            print(f"未找到图片: {image_dir}")
            return None, None

        print(f"\n批量预测: {len(image_files)} 张图片")

        all_results = []
        stats = {
            'total': len(image_files),
            'with_defects': 0,
            'without_defects': 0,
            'class_counts': {name: 0 for name in self.classes_cn}
        }

        for img_path in tqdm(image_files, desc="预测中"):
            result = self.predict_single(img_path, conf=conf)
            all_results.append(result)

            if result['has_defect']:
                stats['with_defects'] += 1
                for name, count in result['class_counts'].items():
                    stats['class_counts'][name] += count
            else:
                stats['without_defects'] += 1

        # 保存结果
        if save_json:
            import json
            json_path = output_dir / 'predictions.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"预测结果已保存: {json_path}")

        # 打印统计
        self._print_batch_stats(stats)

        return all_results, stats

    def _print_batch_stats(self, stats):
        """打印批量预测统计"""
        print("\n" + "=" * 50)
        print("批量预测统计")
        print("=" * 50)
        print(f"总图片: {stats['total']}")
        print(f"有缺陷: {stats['with_defects']} ({100 * stats['with_defects'] / stats['total']:.1f}%)")
        print(f"无缺陷: {stats['without_defects']} ({100 * stats['without_defects'] / stats['total']:.1f}%)")
        print("\n各类别缺陷数量:")
        for name, count in stats['class_counts'].items():
            print(f"  {name}: {count}")

    def visualize(self, image_path, conf=0.25, save_path=None, show=True):
        """
        可视化预测结果

        Args:
            image_path: 图片路径
            conf: 置信度阈值
            save_path: 保存路径
            show: 是否显示
        """
        result = self.predict_single(image_path, conf=conf)

        # 读取图片
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 创建图形
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        # 原图
        axes[0].imshow(img)
        axes[0].set_title('原始图像', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # 检测结果
        img_result = img.copy()

        for det in result['detections']:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            class_id = det['class_id']
            color = self.colors[class_id]

            cv2.rectangle(img_result, (x1, y1), (x2, y2), color, 3)

            label = f"{det['class_name_cn']}: {det['confidence']:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

            if y1 > 30:
                cv2.rectangle(img_result, (x1, y1 - 30), (x1 + w + 10, y1), color, -1)
                cv2.putText(img_result, label, (x1 + 5, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.rectangle(img_result, (x1, y2), (x1 + w + 10, y2 + 30), color, -1)
                cv2.putText(img_result, label, (x1 + 5, y2 + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        axes[1].imshow(img_result)

        if result['has_defect']:
            title = f"检测到 {result['total_defects']} 个缺陷"
        else:
            title = "未检测到缺陷"
        axes[1].set_title(title, fontsize=12, fontweight='bold')
        axes[1].axis('off')

        # 图例
        legend_elements = [
            plt.Line2D([0], [0], color=np.array(self.colors[i]) / 255,
                       linewidth=4, label=f'{self.classes_cn[i]}')
            for i in range(len(self.classes))
        ]
        axes[1].legend(handles=legend_elements, loc='upper right', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"可视化已保存: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return result

    def export(self, weights_path, format='onnx', imgsz=640):
        """
        导出模型

        Args:
            weights_path: 模型权重路径
            format: 导出格式
            imgsz: 输入大小
        """
        self.load_model(weights_path)
        export_path = self.model.export(format=format, imgsz=imgsz)
        print(f"模型已导出: {export_path}")
        return export_path