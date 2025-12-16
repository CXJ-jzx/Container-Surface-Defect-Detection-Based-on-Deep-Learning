# utils/metrics.py
"""
评估指标计算
"""

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, classification_report,
    average_precision_score, roc_auc_score
)


def calculate_metrics(y_true, y_pred, average='macro'):
    """
    计算分类指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        average: 平均方式 ('macro', 'micro', 'weighted')

    Returns:
        指标字典
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }

    return metrics


def calculate_multilabel_metrics(y_true, y_pred, class_names=None):
    """
    计算多标签分类指标

    Args:
        y_true: 真实标签 (N, num_classes)
        y_pred: 预测标签 (N, num_classes)
        class_names: 类别名称列表

    Returns:
        指标字典
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    num_classes = y_true.shape[1]

    if class_names is None:
        class_names = [f'Class_{i}' for i in range(num_classes)]

    metrics = {
        'overall': {
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'exact_match': np.all(y_true == y_pred, axis=1).mean()
        },
        'per_class': {}
    }

    for i, name in enumerate(class_names):
        metrics['per_class'][name] = {
            'precision': precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'recall': recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'f1': f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
            'support': int(y_true[:, i].sum())
        }

    return metrics


def calculate_detection_metrics(predictions, ground_truths, iou_threshold=0.5):
    """
    计算目标检测指标

    Args:
        predictions: 预测结果列表
        ground_truths: 真实标签列表
        iou_threshold: IoU阈值

    Returns:
        指标字典
    """
    # 简化实现，实际使用ultralytics的metrics
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred.get('detections', [])
        gt_boxes = gt.get('boxes', [])

        matched = [False] * len(gt_boxes)

        for pred_box in pred_boxes:
            best_iou = 0
            best_idx = -1

            for idx, gt_box in enumerate(gt_boxes):
                if matched[idx]:
                    continue
                iou = calculate_iou(pred_box['bbox'], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= iou_threshold and best_idx >= 0:
                total_tp += 1
                matched[best_idx] = True
            else:
                total_fp += 1

        total_fn += sum(1 for m in matched if not m)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }


def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU

    Args:
        box1: 边界框1 (dict或list)
        box2: 边界框2 (dict或list)

    Returns:
        IoU值
    """
    if isinstance(box1, dict):
        x1_1, y1_1 = box1['x1'], box1['y1']
        x2_1, y2_1 = box1['x2'], box1['y2']
    else:
        x1_1, y1_1, x2_1, y2_1 = box1[:4]

    if isinstance(box2, dict):
        x1_2, y1_2 = box2['x1'], box2['y1']
        x2_2, y2_2 = box2['x2'], box2['y2']
    else:
        x1_2, y1_2, x2_2, y2_2 = box2[:4]

    # 计算交集
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # 计算并集
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0

    return inter_area / union_area


def print_metrics(metrics, title="评估结果"):
    """打印指标"""
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)

    if 'overall' in metrics:
        print("\n总体指标:")
        for key, value in metrics['overall'].items():
            print(f"  {key}: {value:.4f}")

        if 'per_class' in metrics:
            print("\n各类别指标:")
            for class_name, class_metrics in metrics['per_class'].items():
                print(f"\n  {class_name}:")
                for key, value in class_metrics.items():
                    if isinstance(value, float):
                        print(f"    {key}: {value:.4f}")
                    else:
                        print(f"    {key}: {value}")
    else:
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    print("=" * 50)