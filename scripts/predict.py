# scripts/predict.py
"""
推理预测脚本
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import argparse
import torch
from PIL import Image

from config.config import Config
from data.transforms import get_test_transforms
from models.classifier import MultiLabelClassifier, ClassifierTrainer
from models.detector import ContainerDetector
from utils.visualization import Visualizer


def parse_args():
    parser = argparse.ArgumentParser(description='推理预测')

    parser.add_argument('--source', type=str, required=True,
                        help='输入图片或文件夹路径')
    parser.add_argument('--model', type=str, default='both',
                        choices=['classifier', 'detector', 'both'],
                        help='使用的模型')
    parser.add_argument('--classifier-weights', type=str, default=None,
                        help='分类模型权重路径')
    parser.add_argument('--detector-weights', type=str, default=None,
                        help='检测模型权重路径')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--show', action='store_true',
                        help='显示结果')
    parser.add_argument('--save', action='store_true', default=True,
                        help='保存结果')

    return parser.parse_args()


class InferencePipeline:
    """推理管道"""

    def __init__(self, classifier_weights=None, detector_weights=None, device=None):
        self.device = device or Config.DEVICE
        self.visualizer = Visualizer()

        self.classifier = None
        self.classifier_trainer = None
        self.detector = None

        self.transform = get_test_transforms(Config.CLS_IMAGE_SIZE)

        # 加载分类模型
        if classifier_weights:
            self._load_classifier(classifier_weights)

        # 加载检测模型
        if detector_weights:
            self._load_detector(detector_weights)

    def _load_classifier(self, weights_path):
        """加载分类模型"""
        print(f"加载分类模型: {weights_path}")
        self.classifier = MultiLabelClassifier(
            num_classes=Config.NUM_CLASSES,
            backbone=Config.CLASSIFIER_BACKBONE
        )
        self.classifier_trainer = ClassifierTrainer(
            self.classifier, device=self.device
        )
        self.classifier_trainer.load_model(weights_path)

    def _load_detector(self, weights_path):
        """加载检测模型"""
        print(f"加载检测模型: {weights_path}")
        self.detector = ContainerDetector(device=self.device)
        self.detector.load_model(weights_path)

    def predict(self, image_path, conf=0.25):
        """
        预测单张图片

        Returns:
            综合结果字典
        """
        image_path = Path(image_path)
        result = {
            'image_path': str(image_path),
            'classification': None,
            'detection': None
        }

        # 分类预测
        if self.classifier_trainer:
            image = Image.open(image_path).convert('RGB')
            cls_result = self.classifier_trainer.predict(image, self.transform)
            result['classification'] = cls_result

        # 检测预测
        if self.detector:
            det_result = self.detector.predict_single(image_path, conf=conf)
            result['detection'] = det_result

        return result

    def predict_and_visualize(self, image_path, conf=0.25, save_path=None, show=True):
        """预测并可视化"""
        result = self.predict(image_path, conf=conf)

        # 可视化检测结果
        if self.detector and result['detection']:
            self.detector.visualize(
                image_path, conf=conf,
                save_path=save_path, show=show
            )

        # 打印结果
        self._print_result(result)

        return result

    def _print_result(self, result):
        """打印预测结果"""
        print("\n" + "-" * 50)
        print(f"图片: {Path(result['image_path']).name}")
        print("-" * 50)

        if result['classification']:
            print("\n【分类结果】")
            for name, info in result['classification']['predictions'].items():
                status = "✓" if info['detected'] else "✗"
                print(f"  {status} {info['name_cn']}: {info['probability']:.2%}")

        if result['detection']:
            det = result['detection']
            print(f"\n【检测结果】")
            print(f"  是否有缺陷: {'是' if det['has_defect'] else '否'}")
            print(f"  缺陷数量: {det['total_defects']}")

            if det['detections']:
                print(f"  详细信息:")
                for i, d in enumerate(det['detections'], 1):
                    print(f"    {i}. {d['class_name_cn']} "
                          f"(置信度: {d['confidence']:.2%})")


def main():
    args = parse_args()

    Config.setup()

    # 确定权重路径
    classifier_weights = None
    detector_weights = None

    if args.model in ['classifier', 'both']:
        classifier_weights = (args.classifier_weights
                              if args.classifier_weights
                              else Config.CLASSIFIER_WEIGHTS)
        if not Path(classifier_weights).exists():
            print(f"警告: 分类模型不存在: {classifier_weights}")
            classifier_weights = None

    if args.model in ['detector', 'both']:
        detector_weights = (args.detector_weights
                            if args.detector_weights
                            else Config.DETECTOR_WEIGHTS)
        if not Path(detector_weights).exists():
            print(f"警告: 检测模型不存在: {detector_weights}")
            detector_weights = None

    # 创建推理管道
    pipeline = InferencePipeline(
        classifier_weights=classifier_weights,
        detector_weights=detector_weights
    )

    # 确定输出目录
    output_dir = Path(args.output) if args.output else Config.RESULT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 处理输入
    source = Path(args.source)

    if source.is_file():
        # 单张图片
        save_path = output_dir / f"result_{source.stem}.png" if args.save else None
        pipeline.predict_and_visualize(
            source, conf=args.conf,
            save_path=save_path, show=args.show
        )

    elif source.is_dir():
        # 文件夹
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in extensions:
            image_files.extend(list(source.glob(ext)))
            image_files.extend(list(source.glob(ext.upper())))

        print(f"\n找到 {len(image_files)} 张图片")

        for img_path in image_files:
            save_path = output_dir / f"result_{img_path.stem}.png" if args.save else None
            pipeline.predict_and_visualize(
                img_path, conf=args.conf,
                save_path=save_path, show=args.show
            )
    else:
        print(f"无效的输入: {source}")


if __name__ == '__main__':
    main()