#数据分析

# data/analyze.py
"""
数据集分析工具
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import sys

sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config


class DatasetAnalyzer:
    """数据集分析器"""

    def __init__(self, dataset_path=None):
        """
        Args:
            dataset_path: 数据集路径
        """
        self.dataset_path = Path(dataset_path) if dataset_path else Config.DATASET_PATH
        self.classes = Config.CLASSES
        self.classes_cn = Config.CLASSES_CN
        self.colors = Config.CLASS_COLORS

    def analyze_split(self, split='train'):
        """
        分析数据集某个划分的标签

        Args:
            split: 'train', 'val', 或 'test'

        Returns:
            统计信息字典
        """
        label_dir = self.dataset_path / 'labels' / split

        if not label_dir.exists():
            print(f"标签目录不存在: {label_dir}")
            return None

        stats = {
            'split': split,
            'total_images': 0,
            'images_with_defects': 0,
            'images_without_defects': 0,
            'class_counts': defaultdict(int),
            'defects_per_image': [],
            'bbox_sizes': defaultdict(list),
            'bbox_ratios': defaultdict(list),
            'multi_class_images': 0
        }

        label_files = list(label_dir.glob('*.txt'))
        stats['total_images'] = len(label_files)

        for label_file in tqdm(label_files, desc=f'分析 {split} 集'):
            classes_in_image = set()
            num_defects = 0

            with open(label_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if len(lines) == 0 or all(line.strip() == '' for line in lines):
                stats['images_without_defects'] += 1
                stats['defects_per_image'].append(0)
                continue

            stats['images_with_defects'] += 1

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        class_id = int(parts[0])
                        x_center, y_center = float(parts[1]), float(parts[2])
                        width, height = float(parts[3]), float(parts[4])

                        if class_id < len(self.classes):
                            stats['class_counts'][class_id] += 1
                            classes_in_image.add(class_id)
                            num_defects += 1

                            # 边界框统计
                            area = width * height
                            stats['bbox_sizes'][class_id].append(area)

                            if height > 0:
                                ratio = width / height
                                stats['bbox_ratios'][class_id].append(ratio)
                    except (ValueError, IndexError):
                        continue

            stats['defects_per_image'].append(num_defects)

            if len(classes_in_image) > 1:
                stats['multi_class_images'] += 1

        return stats

    def analyze_all(self):
        """分析所有数据集划分"""
        results = {}

        for split in ['train', 'val', 'test']:
            split_dir = self.dataset_path / 'labels' / split
            if split_dir.exists():
                results[split] = self.analyze_split(split)

        return results

    def print_statistics(self, stats):
        """打印统计信息"""
        if stats is None:
            return

        print(f"\n{'=' * 50}")
        print(f"{stats['split'].upper()} 集统计")
        print('=' * 50)

        print(f"\n图片统计:")
        print(f"  总数: {stats['total_images']}")
        print(f"  有缺陷: {stats['images_with_defects']}")
        print(f"  无缺陷: {stats['images_without_defects']}")
        print(f"  多类别缺陷: {stats['multi_class_images']}")

        print(f"\n类别分布:")
        total_defects = sum(stats['class_counts'].values())
        for i, class_name in enumerate(self.classes_cn):
            count = stats['class_counts'].get(i, 0)
            pct = 100 * count / total_defects if total_defects > 0 else 0
            print(f"  {class_name} ({self.classes[i]}): {count} ({pct:.1f}%)")

        print(f"\n每张图片缺陷数量:")
        if stats['defects_per_image']:
            arr = np.array(stats['defects_per_image'])
            print(f"  平均: {arr.mean():.2f}")
            print(f"  最小: {arr.min()}")
            print(f"  最大: {arr.max()}")
            print(f"  中位数: {np.median(arr):.1f}")

    def visualize_statistics(self, save_path=None):
        """可视化统计结果"""
        results = self.analyze_all()

        if not results:
            print("没有可分析的数据")
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. 各类别样本数量
        ax = axes[0, 0]
        if 'train' in results:
            train_stats = results['train']
            counts = [train_stats['class_counts'].get(i, 0) for i in range(len(self.classes))]
            bars = ax.bar(self.classes_cn, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax.set_title('训练集各类别缺陷数量', fontsize=12, fontweight='bold')
            ax.set_ylabel('数量')
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                        str(count), ha='center', va='bottom', fontsize=10)

        # 2. 训练集/测试集对比
        ax = axes[0, 1]
        x = np.arange(len(self.classes))
        width = 0.35

        if 'train' in results and 'test' in results:
            train_counts = [results['train']['class_counts'].get(i, 0) for i in range(len(self.classes))]
            test_counts = [results['test']['class_counts'].get(i, 0) for i in range(len(self.classes))]

            ax.bar(x - width / 2, train_counts, width, label='训练集', color='#3498db')
            ax.bar(x + width / 2, test_counts, width, label='测试集', color='#e74c3c')
            ax.set_xticks(x)
            ax.set_xticklabels(self.classes_cn)
            ax.set_title('训练集 vs 测试集分布', fontsize=12, fontweight='bold')
            ax.legend()

        # 3. 每张图片缺陷数量分布
        ax = axes[0, 2]
        if 'train' in results:
            ax.hist(results['train']['defects_per_image'], bins=20,
                    color='#9b59b6', edgecolor='white', alpha=0.8)
            ax.set_title('每张图片缺陷数量分布', fontsize=12, fontweight='bold')
            ax.set_xlabel('缺陷数量')
            ax.set_ylabel('图片数量')

        # 4. 边界框大小分布
        ax = axes[1, 0]
        if 'train' in results:
            for i, class_name in enumerate(self.classes_cn):
                if results['train']['bbox_sizes'][i]:
                    ax.hist(results['train']['bbox_sizes'][i], bins=30,
                            alpha=0.6, label=class_name)
            ax.set_title('边界框大小分布', fontsize=12, fontweight='bold')
            ax.set_xlabel('相对面积 (width × height)')
            ax.legend()

        # 5. 边界框宽高比分布
        ax = axes[1, 1]
        if 'train' in results:
            for i, class_name in enumerate(self.classes_cn):
                if results['train']['bbox_ratios'][i]:
                    ax.hist(results['train']['bbox_ratios'][i], bins=30,
                            alpha=0.6, label=class_name)
            ax.set_title('边界框宽高比分布', fontsize=12, fontweight='bold')
            ax.set_xlabel('宽高比 (width / height)')
            ax.legend()

        # 6. 数据集摘要表格
        ax = axes[1, 2]
        ax.axis('off')

        if 'train' in results:
            train_stats = results['train']
            test_stats = results.get('test', {})

            table_data = [
                ['指标', '训练集', '测试集'],
                ['总图片', str(train_stats['total_images']),
                 str(test_stats.get('total_images', '-'))],
                ['有缺陷', str(train_stats['images_with_defects']),
                 str(test_stats.get('images_with_defects', '-'))],
                ['凹陷', str(train_stats['class_counts'].get(0, 0)),
                 str(test_stats.get('class_counts', {}).get(0, '-'))],
                ['破洞', str(train_stats['class_counts'].get(1, 0)),
                 str(test_stats.get('class_counts', {}).get(1, '-'))],
                ['锈蚀', str(train_stats['class_counts'].get(2, 0)),
                 str(test_stats.get('class_counts', {}).get(2, '-'))]
            ]

            table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                             colWidths=[0.3, 0.3, 0.3])
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1.2, 1.8)

            # 设置表头样式
            for i in range(3):
                table[(0, i)].set_facecolor('#4472C4')
                table[(0, i)].set_text_props(color='white', fontweight='bold')

            ax.set_title('数据集摘要', fontsize=12, fontweight='bold', pad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"统计图已保存: {save_path}")

        plt.show()

        # 打印详细统计
        for split, stats in results.items():
            self.print_statistics(stats)

    def visualize_samples(self, split='train', num_samples=9, save_path=None):
        """可视化样本图片"""
        image_dir = self.dataset_path / 'images' / split
        label_dir = self.dataset_path / 'labels' / split

        if not image_dir.exists():
            print(f"图片目录不存在: {image_dir}")
            return

        # 获取图片列表
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))

        if not image_files:
            print(f"未找到图片文件")
            return

        # 随机选择样本
        np.random.seed(42)
        indices = np.random.choice(len(image_files),
                                   min(num_samples, len(image_files)),
                                   replace=False)

        # 创建图形
        cols = 3
        rows = (num_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten()

        for idx, ax in enumerate(axes):
            if idx >= len(indices):
                ax.axis('off')
                continue

            img_path = image_files[indices[idx]]
            label_path = label_dir / (img_path.stem + '.txt')

            # 读取图片
            img = cv2.imread(str(img_path))
            if img is None:
                ax.axis('off')
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]

            # 读取并绘制标注
            defect_info = []
            if label_path.exists():
                with open(label_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                class_id = int(parts[0])
                                x_center, y_center = float(parts[1]), float(parts[2])
                                bw, bh = float(parts[3]), float(parts[4])

                                # 转换为像素坐标
                                x1 = int((x_center - bw / 2) * w)
                                y1 = int((y_center - bh / 2) * h)
                                x2 = int((x_center + bw / 2) * w)
                                y2 = int((y_center + bh / 2) * h)

                                # 绘制边界框
                                color = self.colors.get(class_id, (128, 128, 128))
                                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

                                # 标签
                                label = self.classes[class_id]
                                cv2.putText(img, label, (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                                defect_info.append(self.classes_cn[class_id])
                            except (ValueError, IndexError):
                                continue

            ax.imshow(img)
            title = f"{img_path.name}"
            if defect_info:
                title += f"\n{', '.join(defect_info)}"
            ax.set_title(title, fontsize=9)
            ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"样本可视化已保存: {save_path}")

        plt.show()


def main():
    """测试数据分析功能"""
    Config.setup()

    analyzer = DatasetAnalyzer()

    # 可视化统计
    save_path = Config.FIGURE_DIR / 'dataset_analysis.png'
    analyzer.visualize_statistics(save_path=save_path)

    # 可视化样本
    save_path = Config.FIGURE_DIR / 'sample_visualization.png'
    analyzer.visualize_samples(split='train', num_samples=9, save_path=save_path)


if __name__ == '__main__':
    main()