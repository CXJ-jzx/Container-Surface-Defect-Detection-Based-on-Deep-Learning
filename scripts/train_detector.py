# scripts/train_detector.py
"""
训练检测模型脚本
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import argparse

from config.config import Config
from models.detector import ContainerDetector, create_dataset_yaml
from data.analyze import DatasetAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(description='训练YOLO检测模型')

    parser.add_argument('--data', type=str, default=None,
                        help='数据集路径')
    parser.add_argument('--model-size', type=str, default='m',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='模型大小')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--image-size', type=int, default=640,
                        help='输入图像大小')
    parser.add_argument('--resume', action='store_true',
                        help='恢复训练')
    parser.add_argument('--analyze', action='store_true',
                        help='训练前分析数据集')

    return parser.parse_args()


def main():
    args = parse_args()

    # 初始化配置
    Config.setup()
    Config.print_config()

    dataset_path = args.data if args.data else Config.DATASET_PATH

    print("\n" + "=" * 60)
    print("训练YOLO检测模型")
    print("=" * 60)
    print(f"数据集: {dataset_path}")
    print(f"模型: YOLOv8{args.model_size}")
    print(f"轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"图像大小: {args.image_size}")

    # 数据分析
    if args.analyze:
        print("\n分析数据集...")
        analyzer = DatasetAnalyzer(dataset_path)
        analyzer.visualize_statistics(
            save_path=Config.FIGURE_DIR / 'dataset_analysis.png'
        )

    # 创建数据集配置文件
    print("\n创建数据集配置...")
    yaml_path = create_dataset_yaml(
        dataset_path=dataset_path,
        output_path=Config.CONFIG_DIR
    )

    # 创建检测器
    detector = ContainerDetector(
        model_size=args.model_size,
        device=Config.DEVICE
    )

    # 训练
    results = detector.train(
        data_yaml=yaml_path,
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        resume=args.resume
    )

    # 验证
    print("\n在测试集上验证...")
    detector.validate(
        data_yaml=yaml_path,
        weights_path=Config.MODEL_DIR / 'best_detector.pt'
    )

    print("\n训练完成!")
    print(f"最佳模型保存在: {Config.MODEL_DIR / 'best_detector.pt'}")


if __name__ == '__main__':
    main()