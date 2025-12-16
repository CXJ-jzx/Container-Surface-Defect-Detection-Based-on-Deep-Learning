# scripts/train_classifier.py
"""
训练分类模型脚本
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import argparse
import torch

from config.config import Config
from data.dataset import get_data_loaders
from models.classifier import MultiLabelClassifier, ClassifierTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='训练多标签分类模型')

    parser.add_argument('--data', type=str, default=None,
                        help='数据集路径')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'efficientnet_b0'],
                        help='骨干网络')
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--image-size', type=int, default=224,
                        help='输入图像大小')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--early-stopping', type=int, default=10,
                        help='早停轮数')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的模型路径')

    return parser.parse_args()


def main():
    args = parse_args()

    # 初始化配置
    Config.setup()
    Config.print_config()

    # 设置参数
    dataset_path = args.data if args.data else Config.DATASET_PATH

    print("\n" + "=" * 60)
    print("训练多标签分类模型")
    print("=" * 60)
    print(f"数据集: {dataset_path}")
    print(f"骨干网络: {args.backbone}")
    print(f"轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"图像大小: {args.image_size}")

    # 获取数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset_path=dataset_path,
        batch_size=args.batch_size,
        image_size=args.image_size,
        val_split=args.val_split,
        num_workers=Config.NUM_WORKERS
    )

    # 创建模型
    model = MultiLabelClassifier(
        num_classes=Config.NUM_CLASSES,
        backbone=args.backbone,
        pretrained=True
    )

    # 如果恢复训练
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"从 {args.resume} 恢复训练")

    # 创建训练器
    trainer = ClassifierTrainer(
        model=model,
        device=Config.DEVICE,
        save_dir=Config.MODEL_DIR
    )

    # 训练
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        early_stopping=args.early_stopping
    )

    # 评估
    print("\n在测试集上评估...")
    best_model_path = Config.MODEL_DIR / 'best_classifier.pth'
    trainer.evaluate(test_loader, model_path=best_model_path)

    print("\n训练完成!")
    print(f"最佳模型保存在: {best_model_path}")


if __name__ == '__main__':
    main()