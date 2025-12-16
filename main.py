# main.py
"""
主程序入口
提供命令行接口统一管理所有功能
"""

import sys
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='集装箱缺陷检测系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
=========

1. 数据分析:
   python main.py analyze --data ./dataset

2. 训练分类模型:
   python main.py train-cls --epochs 30 --batch-size 32

3. 训练检测模型:
   python main.py train-det --epochs 100 --batch-size 16

4. 评估模型:
   python main.py evaluate --model both

5. 预测:
   python main.py predict --source ./test.jpg --show

6. 批量预测:
   python main.py predict --source ./dataset/images/test --save
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 数据分析命令
    analyze_parser = subparsers.add_parser('analyze', help='分析数据集')
    analyze_parser.add_argument('--data', type=str, default='./dataset', help='数据集路径')
    analyze_parser.add_argument('--save', action='store_true', help='保存分析结果')

    # 训练分类模型命令
    train_cls_parser = subparsers.add_parser('train-cls', help='训练分类模型')
    train_cls_parser.add_argument('--data', type=str, default=None, help='数据集路径')
    train_cls_parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    train_cls_parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    train_cls_parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    train_cls_parser.add_argument('--backbone', type=str, default='resnet50', help='骨干网络')

    # 训练检测模型命令
    train_det_parser = subparsers.add_parser('train-det', help='训练检测模型')
    train_det_parser.add_argument('--data', type=str, default=None, help='数据集路径')
    train_det_parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    train_det_parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    train_det_parser.add_argument('--model-size', type=str, default='m', help='模型大小')
    train_det_parser.add_argument('--image-size', type=int, default=640, help='图像大小')

    # 评估命令
    eval_parser = subparsers.add_parser('evaluate', help='评估模型')
    eval_parser.add_argument('--model', type=str, default='both',
                             choices=['classifier', 'detector', 'both'], help='模型类型')
    eval_parser.add_argument('--data', type=str, default=None, help='数据集路径')
    eval_parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')

    # 预测命令
    predict_parser = subparsers.add_parser('predict', help='推理预测')
    predict_parser.add_argument('--source', type=str, required=True, help='输入图片或文件夹')
    predict_parser.add_argument('--model', type=str, default='detector',
                                choices=['classifier', 'detector', 'both'], help='模型类型')
    predict_parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    predict_parser.add_argument('--output', type=str, default=None, help='输出目录')
    predict_parser.add_argument('--show', action='store_true', help='显示结果')
    predict_parser.add_argument('--save', action='store_true', default=True, help='保存结果')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # 执行对应命令
    if args.command == 'analyze':
        from scripts.train_classifier import main as analyze_main
        from config.config import Config
        from data.analyze import DatasetAnalyzer

        Config.setup()
        analyzer = DatasetAnalyzer(args.data)
        save_path = Config.FIGURE_DIR / 'dataset_analysis.png' if args.save else None
        analyzer.visualize_statistics(save_path=save_path)
        analyzer.visualize_samples(save_path=Config.FIGURE_DIR / 'samples.png' if args.save else None)

    elif args.command == 'train-cls':
        sys.argv = ['train_classifier.py',
                    f'--data={args.data}' if args.data else '',
                    f'--epochs={args.epochs}',
                    f'--batch-size={args.batch_size}',
                    f'--lr={args.lr}',
                    f'--backbone={args.backbone}']
        sys.argv = [a for a in sys.argv if a]  # 移除空参数

        from scripts.train_classifier import main
        main()

    elif args.command == 'train-det':
        sys.argv = ['train_detector.py',
                    f'--data={args.data}' if args.data else '',
                    f'--epochs={args.epochs}',
                    f'--batch-size={args.batch_size}',
                    f'--model-size={args.model_size}',
                    f'--image-size={args.image_size}']
        sys.argv = [a for a in sys.argv if a]

        from scripts.train_detector import main
        main()

    elif args.command == 'evaluate':
        sys.argv = ['evaluate.py',
                    f'--model={args.model}',
                    f'--data={args.data}' if args.data else '',
                    f'--conf={args.conf}']
        sys.argv = [a for a in sys.argv if a]

        from scripts.evaluate import main
        main()

    elif args.command == 'predict':
        sys.argv = ['predict.py',
                    f'--source={args.source}',
                    f'--model={args.model}',
                    f'--conf={args.conf}']
        if args.output:
            sys.argv.append(f'--output={args.output}')
        if args.show:
            sys.argv.append('--show')
        if args.save:
            sys.argv.append('--save')

        from scripts.predict import main
        main()


if __name__ == '__main__':
    main()