# utils/visualization.py
"""
可视化工具
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

import sys

sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config


class Visualizer:
    """可视化工具类"""

    def __init__(self):
        self.classes = Config.CLASSES
        self.classes_cn = Config.CLASSES_CN
        self.colors = Config.CLASS_COLORS
        self.save_dir = Config.FIGURE_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def draw_detections(self, image, detections, show=True, save_path=None):
        """
        在图像上绘制检测结果

        Args:
            image: 图像 (numpy array, BGR或RGB)
            detections: 检测结果列表
            show: 是否显示
            save_path: 保存路径

        Returns:
            绘制后的图像
        """
        img = image.copy()

        # 确保是RGB格式
        if len(img.shape) == 3 and img.shape[2] == 3:
            # 假设输入可能是BGR，转换为RGB用于matplotlib显示
            pass

        for det in detections:
            bbox = det['bbox']

            if isinstance(bbox, dict):
                x1, y1 = bbox['x1'], bbox['y1']
                x2, y2 = bbox['x2'], bbox['y2']
            else:
                x1, y1, x2, y2 = bbox[:4]

            class_id = det['class_id']
            confidence = det['confidence']
            class_name_cn = det.get('class_name_cn', self.classes_cn[class_id])

            color = self.colors[class_id]

            # 绘制边界框
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

            # 绘制标签
            label = f"{class_name_cn}: {confidence:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )

            # 标签背景
            if y1 > 30:
                label_y1 = int(y1) - 30
                label_y2 = int(y1)
                text_y = int(y1) - 8
            else:
                label_y1 = int(y2)
                label_y2 = int(y2) + 30
                text_y = int(y2) + 22

            cv2.rectangle(img, (int(x1), label_y1),
                          (int(x1) + text_width + 10, label_y2), color, -1)
            cv2.putText(img, label, (int(x1) + 5, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if save_path:
            # 保存为BGR格式
            cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"图像已保存: {save_path}")

        if show:
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.tight_layout()
            plt.show()

        return img

    def compare_images(self, original, result, title1='原图', title2='检测结果',
                       save_path=None, show=True):
        """
        对比显示原图和检测结果

        Args:
            original: 原始图像
            result: 检测结果图像
            title1: 左图标题
            title2: 右图标题
            save_path: 保存路径
            show: 是否显示
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        axes[0].imshow(original)
        axes[0].set_title(title1, fontsize=14, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(result)
        axes[1].set_title(title2, fontsize=14, fontweight='bold')
        axes[1].axis('off')

        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], color=np.array(self.colors[i]) / 255,
                       linewidth=4, label=f'{self.classes_cn[i]}')
            for i in range(len(self.classes))
        ]
        axes[1].legend(handles=legend_elements, loc='upper right', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"对比图已保存: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_detection_results(self, results, save_path=None, show=True):
        """
        绘制多张图片的检测结果

        Args:
            results: 检测结果列表
            save_path: 保存路径
            show: 是否显示
        """
        n = len(results)
        if n == 0:
            print("没有结果可显示")
            return

        cols = min(3, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, (ax, result) in enumerate(zip(axes, results)):
            # 读取图像
            img_path = result['image_path']
            img = cv2.imread(img_path)

            if img is None:
                ax.axis('off')
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 绘制检测框
            for det in result['detections']:
                bbox = det['bbox']
                x1, y1 = bbox['x1'], bbox['y1']
                x2, y2 = bbox['x2'], bbox['y2']
                class_id = det['class_id']
                color = self.colors[class_id]

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            ax.imshow(img)

            # 设置标题
            defect_count = result['total_defects']
            title = f"缺陷: {defect_count}" if defect_count > 0 else "无缺陷"
            ax.set_title(title, fontsize=10)
            ax.axis('off')

        # 隐藏空白子图
        for idx in range(len(results), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"结果图已保存: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_class_distribution(self, class_counts, title='类别分布',
                                save_path=None, show=True):
        """
        绘制类别分布柱状图

        Args:
            class_counts: 类别计数字典
            title: 图表标题
            save_path: 保存路径
            show: 是否显示
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        names = list(class_counts.keys())
        counts = list(class_counts.values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(names)]

        bars = ax.bar(names, counts, color=colors, edgecolor='white', linewidth=1.5)

        # 添加数值标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                    f'{int(count)}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_xlabel('缺陷类型', fontsize=12)
        ax.set_ylabel('数量', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"分布图已保存: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_confidence_distribution(self, confidences, save_path=None, show=True):
        """
        绘制置信度分布直方图

        Args:
            confidences: 置信度列表
            save_path: 保存路径
            show: 是否显示
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(confidences, bins=20, color='#3498db', edgecolor='white', alpha=0.8)

        # 添加均值线
        mean_conf = np.mean(confidences)
        ax.axvline(mean_conf, color='#e74c3c', linestyle='--', linewidth=2,
                   label=f'平均值: {mean_conf:.3f}')

        ax.set_xlabel('置信度', fontsize=12)
        ax.set_ylabel('数量', fontsize=12)
        ax.set_title('检测置信度分布', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"置信度分布图已保存: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def create_summary_figure(self, stats, save_path=None, show=True):
        """
        创建检测结果汇总图

        Args:
            stats: 统计信息字典
            save_path: 保存路径
            show: 是否显示
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. 有/无缺陷饼图
        ax = axes[0, 0]
        sizes = [stats['with_defects'], stats['without_defects']]
        labels = ['有缺陷', '无缺陷']
        colors_pie = ['#e74c3c', '#2ecc71']
        explode = (0.05, 0)

        ax.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
               autopct='%1.1f%%', shadow=True, startangle=90)
        ax.set_title('缺陷图片比例', fontsize=12, fontweight='bold')

        # 2. 各类别数量柱状图
        ax = axes[0, 1]
        class_counts = stats['class_counts']
        names = list(class_counts.keys())
        counts = list(class_counts.values())
        colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1']

        bars = ax.bar(names, counts, color=colors_bar)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                    str(int(count)), ha='center', va='bottom', fontsize=11)
        ax.set_title('各类别缺陷数量', fontsize=12, fontweight='bold')
        ax.set_ylabel('数量')

        # 3. 统计信息表格
        ax = axes[1, 0]
        ax.axis('off')

        table_data = [
            ['指标', '数值'],
            ['总图片数', str(stats['total'])],
            ['有缺陷图片', str(stats['with_defects'])],
            ['无缺陷图片', str(stats['without_defects'])],
            ['缺陷检出率', f"{100 * stats['with_defects'] / stats['total']:.1f}%"],
        ]

        # 添加各类别
        for name, count in class_counts.items():
            table_data.append([f'{name}数量', str(count)])

        table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                         colWidths=[0.4, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)

        # 设置表头样式
        for i in range(2):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(color='white', fontweight='bold')

        ax.set_title('统计摘要', fontsize=12, fontweight='bold', pad=20)

        # 4. 类别占比饼图
        ax = axes[1, 1]
        if sum(counts) > 0:
            ax.pie(counts, labels=names, colors=colors_bar,
                   autopct='%1.1f%%', shadow=True, startangle=90)
            ax.set_title('各类别缺陷占比', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, '无缺陷数据', ha='center', va='center', fontsize=14)
            ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"汇总图已保存: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()