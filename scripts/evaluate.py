# scripts/evaluate.py
"""
模型评估脚本
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import argparse
import torch

from config.config import Config
from data.dataset import get_data_loaders
from models.classifier import MultiLabelClassifier, ClassifierTrainer
from models.detector import ContainerDetector


def parse_args():
    parser = argparse.ArgumentParser(description='评估模型')

    parser.add_argument('--model', type=str, required=True,
                        choices=['classifier', 'detector', 'both'],
                        help='要评估的模型类型')
    parser.add_argument('--classifier-weights', type=str, default=None,
                        help='分类模型权重路径')
    parser.add_argument('--detector-weights', type=str, default=None,
                        help='检测模型权重路径')
    parser.add_argument('--data', type=str, default=None,
                        help='数据集路径')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='检测置信度阈值')

    return parser.parse_args()


def evaluate_classifier(args):
    """评估分类模型"""
    print("\n" + "=" * 60)
    print("评估分类模型")
    print("=" * 60)

    dataset_path = args.data if args.data else Config.DATASET_PATH
    weights_path = args.classifier_weights if args.classifier_weights else Config.CLASSIFIER_WEIGHTS

    # 加载数据
    _, _, test_loader = get_data_loaders(
        dataset_path=dataset_path,
        batch_size=args.batch_size
    )

    # 加载模型
    model = MultiLabelClassifier(
        num_classes=Config.NUM_CLASSES,
        backbone=Config.CLASSIFIER_BACKBONE
    )

    trainer = ClassifierTrainer(model, device=Config.DEVICE)
    trainer.evaluate(test_loader, model_path=weights_path)


def evaluate_detector(args):
    """评估检测模型"""
    print("\n" + "=" * 60)
    print("评估检测模型")
    print("=" * 60)

    dataset_path = args.data if args.data else Config.DATASET_PATH
    weights_path = args.detector_weights if args.detector_weights else Config.DETECTOR_WEIGHTS

    # 创建检测器
    detector = ContainerDetector(device=Config.DEVICE)

    # 验证
    yaml_path = Config.DATASET_YAML
    if not yaml_path.exists():
        from models.detector import create_dataset_yaml
        yaml_path = create_dataset_yaml(dataset_path)

    detector.validate(
        data_yaml=str(yaml_path),
        weights_path=weights_path,
        conf=args.conf
    )


def main():
    args = parse_args()

    Config.setup()

    if args.model == 'classifier':
        evaluate_classifier(args)
    elif args.model == 'detector':
        evaluate_detector(args)
    elif args.model == 'both':
        evaluate_classifier(args)
        evaluate_detector(args)


if __name__ == '__main__':
    main()