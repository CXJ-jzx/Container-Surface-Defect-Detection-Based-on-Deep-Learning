# Container-Surface-Defect-Detection-Based-on-Deep-Learning
基于深度学习的集装箱表面缺陷检测系统，支持三类缺陷的自动识别与精确定位。

# 🚢 集装箱缺陷检测系统

基于深度学习的集装箱表面缺陷检测系统，支持三类缺陷的自动识别与精确定位。

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 目录

- [项目简介](#项目简介)
- [功能特性](#功能特性)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [数据集准备](#数据集准备)
- [快速开始](#快速开始)
- [详细使用说明](#详细使用说明)
- [模型说明](#模型说明)
- [配置参数](#配置参数)
- [性能指标](#性能指标)
- [常见问题](#常见问题)
- [更新日志](#更新日志)
- [许可证](#许可证)

---

## 📖 项目简介

本项目针对集装箱表面缺陷检测任务，实现了一套完整的深度学习检测系统。系统能够自动识别和定位集装箱表面的三类常见缺陷：

| 缺陷类型 | 英文名称 | 描述 |
|---------|---------|------|
| 🔴 凹陷 | dent | 集装箱表面因碰撞产生的凹陷变形 |
| 🟢 破洞 | hole | 集装箱表面的穿透性破损 |
| 🔵 锈蚀 | rusty | 集装箱表面的氧化锈蚀区域 |

### 任务定义

1. **分类任务**：判断图像中是否存在缺陷，以及存在哪些类型的缺陷
2. **检测任务**：定位缺陷的具体位置（边界框），并识别缺陷类别

---

## ✨ 功能特性

- ✅ **多标签分类**：判断图像中存在的缺陷类型（支持同时存在多种缺陷）
- ✅ **目标检测**：精确定位每个缺陷的位置和类别
- ✅ **数据分析**：自动分析数据集分布，生成可视化报告
- ✅ **模型训练**：支持从预训练模型微调，提供完整训练流程
- ✅ **批量预测**：支持单张图片和批量文件夹预测
- ✅ **可视化结果**：自动生成带标注的检测结果图
- ✅ **模型导出**：支持导出为ONNX等格式，便于部署
- ✅ **命令行接口**：提供统一的命令行工具，操作简便

---

## 📁 项目结构

```text
ContainerDefectDetection/
│
├── config/ # 配置文件目录
│ ├── init.py
│ ├── config.py # 全局配置参数
│ └── container_dataset.yaml # 数据集配置（自动生成）
│
├── data/ # 数据处理模块
│ ├── init.py
│ ├── dataset.py # 数据集类定义
│ ├── transforms.py # 数据增强方法
│ └── analyze.py # 数据集分析工具
│
├── models/ # 模型定义模块
│ ├── init.py
│ ├── classifier.py # 多标签分类模型
│ └── detector.py # YOLO检测模型
│
├── utils/ # 工具函数模块
│ ├── init.py
│ ├── visualization.py # 可视化工具
│ ├── metrics.py # 评估指标计算
│ └── helpers.py # 辅助函数
│
├── scripts/ # 运行脚本
│ ├── train_classifier.py # 训练分类模型
│ ├── train_detector.py # 训练检测模型
│ ├── evaluate.py # 模型评估
│ └── predict.py # 推理预测
│
├── dataset/ # 数据集目录
│ ├── images/
│ │ ├── train/ # 训练集图片
│ │ ├── val/ # 验证集图片（可选）
│ │ └── test/ # 测试集图片
│ └── labels/
│ ├── train/ # 训练集标签（YOLO格式）
│ ├── val/ # 验证集标签
│ └── test/ # 测试集标签
│
├── outputs/ # 输出目录
│ ├── models/ # 保存的模型权重
│ ├── logs/ # 训练日志
│ ├── results/ # 预测结果
│ └── figures/ # 可视化图表
│
├── main.py # 主程序入口
├── requirements.txt # Python依赖包
└── README.md # 项目说明文档
```

## 🚀 快速开始
方式一：使用主程序入口（推荐）

```bash
# 1. 分析数据集
python main.py analyze --data ./dataset --save

# 2. 训练检测模型（推荐优先使用）
python main.py train-det --epochs 100 --batch-size 16

# 3. 训练分类模型（可选）
python main.py train-cls --epochs 30 --batch-size 32

# 4. 评估模型
python main.py evaluate --model detector

# 5. 预测
python main.py predict --source ./test_image.jpg --show
```
