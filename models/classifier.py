# models/classifier.py
"""
多标签分类模型
用于判断图片中存在哪些类型的缺陷
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns

import sys

sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config


class MultiLabelClassifier(nn.Module):
    """
    多标签分类模型
    基于预训练的骨干网络
    """

    def __init__(self, num_classes=3, backbone='resnet50', pretrained=True, dropout=0.5):
        """
        Args:
            num_classes: 类别数量
            backbone: 骨干网络 ('resnet50', 'resnet101', 'efficientnet_b0')
            pretrained: 是否使用预训练权重
            dropout: Dropout比率
        """
        super().__init__()

        self.num_classes = num_classes
        self.backbone_name = backbone

        # 选择骨干网络
        if backbone == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = self._create_classifier(in_features, num_classes, dropout)

        elif backbone == 'resnet101':
            weights = models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet101(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = self._create_classifier(in_features, num_classes, dropout)

        elif backbone == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = self._create_classifier(in_features, num_classes, dropout)

        else:
            raise ValueError(f"不支持的骨干网络: {backbone}")

    def _create_classifier(self, in_features, num_classes, dropout):
        """创建分类头"""
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
            nn.Sigmoid()  # 多标签分类使用Sigmoid
        )

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self, freeze=True):
        """冻结/解冻骨干网络"""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

        # 保持分类头可训练
        if self.backbone_name.startswith('resnet'):
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        else:
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True


class ClassifierTrainer:
    """分类模型训练器"""

    def __init__(self, model, device=None, save_dir=None):
        """
        Args:
            model: 分类模型
            device: 计算设备
            save_dir: 模型保存目录
        """
        self.device = device or Config.DEVICE
        self.model = model.to(self.device)
        self.save_dir = Path(save_dir) if save_dir else Config.MODEL_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.classes = Config.CLASSES
        self.classes_cn = Config.CLASSES_CN

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_f1': [],
            'val_f1': []
        }

    def train(self, train_loader, val_loader, epochs=30, lr=1e-4,
              weight_decay=1e-4, early_stopping=10):
        """
        训练模型

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            lr: 学习率
            weight_decay: 权重衰减
            early_stopping: 早停轮数

        Returns:
            训练历史
        """
        print("\n" + "=" * 60)
        print("开始训练分类模型")
        print("=" * 60)
        print(f"设备: {self.device}")
        print(f"轮数: {epochs}")
        print(f"学习率: {lr}")

        # 损失函数和优化器
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        best_f1 = 0
        patience_counter = 0

        for epoch in range(epochs):
            # 训练阶段
            train_loss, train_f1 = self._train_epoch(
                train_loader, criterion, optimizer
            )

            # 验证阶段
            val_loss, val_f1 = self._validate_epoch(val_loader, criterion)

            # 更新学习率
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)

            # 打印进度
            print(f"\nEpoch [{epoch + 1}/{epochs}]")
            print(f"  Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
            print(f"  LR: {current_lr:.6f}")

            # 保存最佳模型
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                self._save_model('best_classifier.pth')
                print(f"  ✓ 保存最佳模型 (F1: {best_f1:.4f})")
            else:
                patience_counter += 1
                print(f"  早停计数: {patience_counter}/{early_stopping}")

            # 早停检查
            if patience_counter >= early_stopping:
                print(f"\n早停触发，训练结束")
                break

        # 保存最后模型
        self._save_model('last_classifier.pth')

        # 绘制训练曲线
        self._plot_training_history()

        print(f"\n训练完成! 最佳 F1: {best_f1:.4f}")

        return self.history

    def _train_epoch(self, loader, criterion, optimizer):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(loader, desc='Training')
        for images, labels, _ in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = (outputs > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 计算F1
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        return total_loss / len(loader), f1

    def _validate_epoch(self, loader, criterion):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels, _ in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item()

                preds = (outputs > 0.5).float()
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        return total_loss / len(loader), f1

    def evaluate(self, test_loader, model_path=None):
        """
        评估模型

        Args:
            test_loader: 测试数据加载器
            model_path: 模型路径（可选）
        """
        if model_path:
            self.load_model(model_path)

        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels, _ in tqdm(test_loader, desc='Evaluating'):
                images = images.to(self.device)
                outputs = self.model(images)
                preds = (outputs > 0.5).float()

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.numpy())

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        # 打印分类报告
        print("\n" + "=" * 60)
        print("分类评估报告")
        print("=" * 60)

        for i, (name, name_cn) in enumerate(zip(self.classes, self.classes_cn)):
            print(f"\n【{name_cn} ({name})】")
            print(classification_report(
                all_labels[:, i],
                all_preds[:, i],
                target_names=['无', '有'],
                zero_division=0
            ))

        # 绘制混淆矩阵
        self._plot_confusion_matrices(all_labels, all_preds)

        # 总体指标
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)

        print(f"\n总体指标:")
        print(f"  Macro F1: {f1_macro:.4f}")
        print(f"  Micro F1: {f1_micro:.4f}")

        return all_preds, all_labels

    def _plot_confusion_matrices(self, labels, preds):
        """绘制各类别混淆矩阵"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for i, (ax, name_cn) in enumerate(zip(axes, self.classes_cn)):
            cm = confusion_matrix(labels[:, i], preds[:, i])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['无', '有'],
                        yticklabels=['无', '有'])
            ax.set_xlabel('预测')
            ax.set_ylabel('真实')
            ax.set_title(f'{name_cn}')

        plt.suptitle('各类别混淆矩阵', fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = self.save_dir.parent / 'figures' / 'classifier_confusion_matrix.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"混淆矩阵已保存: {save_path}")

    def _plot_training_history(self):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 损失曲线
        axes[0].plot(self.history['train_loss'], label='Train', color='#3498db')
        axes[0].plot(self.history['val_loss'], label='Validation', color='#e74c3c')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('训练/验证损失', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # F1曲线
        axes[1].plot(self.history['train_f1'], label='Train', color='#3498db')
        axes[1].plot(self.history['val_f1'], label='Validation', color='#e74c3c')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_title('训练/验证 F1 分数', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = self.save_dir.parent / 'figures' / 'classifier_training_history.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"训练曲线已保存: {save_path}")

    def _save_model(self, filename):
        """保存模型"""
        path = self.save_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.model.num_classes,
            'backbone': self.model.backbone_name,
            'history': self.history
        }, path)

    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"模型已加载: {path}")

    def predict(self, image, transform=None):
        """
        预测单张图片

        Args:
            image: PIL Image 或 tensor
            transform: 预处理变换

        Returns:
            预测结果字典
        """
        self.model.eval()

        if transform and not isinstance(image, torch.Tensor):
            image = transform(image)

        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        with torch.no_grad():
            output = self.model(image)
            probs = output[0].cpu().numpy()
            preds = (probs > 0.5).astype(int)

        result = {
            'predictions': {},
            'has_defect': preds.sum() > 0
        }

        for i, (name, name_cn) in enumerate(zip(self.classes, self.classes_cn)):
            result['predictions'][name] = {
                'detected': bool(preds[i]),
                'probability': float(probs[i]),
                'name_cn': name_cn
            }

        return result