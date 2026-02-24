# -*- coding: utf-8 -*-
"""
文件名: visualize_results.py
项目: retina_ph_seg
功能: 可视化展示模块（对比图+测试集预测+训练曲线）
作者: AI Assistant
创建日期: 2026-02-07
Python版本: 3.12+

依赖关系:
  - 上游: config.yaml, data_drive.py, model_unet.py
  - 下游: 无（独立可视化脚本）

主要功能:
  - 展示验证集随机样本对比图（原图、ROI掩码、金标、预测）
  - 展示测试集随机样本预测结果（原图、ROI掩码、预测）
  - 生成训练曲线图（2×3布局，6个指标）
  - 基于最佳模型进行推理

更新记录:
  - v1.0: 初始版本，实现对比图可视化
  - v1.1: 添加训练曲线生成功能（2×3布局，6个指标）
"""

import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm

# 强制重新构建字体缓存
fm._load_fontmanager(try_read_cache=False)

# 设置支持中文的字体
chinese_fonts = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'Noto Serif CJK JP', 
                 'SimHei', 'WenQuanYi Micro Hei', 'AR PL UMing CN']
available_fonts = [f.name for f in fm.fontManager.ttflist]

for font_name in chinese_fonts:
    if font_name in available_fonts:
        matplotlib.rcParams['font.sans-serif'] = [font_name] + matplotlib.rcParams['font.sans-serif']
        matplotlib.rcParams['axes.unicode_minus'] = False
        print(f'使用字体: {font_name}')
        break

import matplotlib.pyplot as plt
import yaml

from data_drive import DRIVEDataset
from model_unet import load_model


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """将PyTorch张量转换为NumPy数组。"""
    if tensor.is_cuda:
        tensor = tensor.cpu()
    array = tensor.detach().numpy()
    if array.ndim == 4:
        array = array[0, 0]
    elif array.ndim == 3:
        array = array[0]
    return array


def visualize_val_sample(
    model: torch.nn.Module,
    dataset: DRIVEDataset,
    device: str = 'cuda',
    sample_idx: Optional[int] = None,
    save_path: str = 'results/val_sample_comparison.png'
) -> None:
    """可视化验证集单个样本的对比图。
    
    显示4张图：原图、ROI掩码、金标、预测
    
    Args:
        model: 已加载的模型
        dataset: 验证集数据集
        device: 计算设备
        sample_idx: 样本索引（None则随机选择）
        save_path: 保存路径
    """
    model.eval()
    
    # 随机选择样本
    if sample_idx is None:
        sample_idx = random.randint(0, len(dataset) - 1)
    
    img_id = dataset.image_ids[sample_idx]
    image, vessel, roi = dataset[sample_idx]
    
    # 预测
    with torch.no_grad():
        image_batch = image.unsqueeze(0).to(device)
        output = model(image_batch)
        pred = torch.sigmoid(output[0, 0]).cpu()
    
    # 转为numpy
    image_np = tensor_to_numpy(image)
    vessel_np = tensor_to_numpy(vessel)
    roi_np = tensor_to_numpy(roi)
    pred_np = tensor_to_numpy(pred)
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f'验证集样本对比图 (ID: {img_id:02d})', fontsize=14)
    
    # 1. 原图（绿色通道）
    axes[0, 0].imshow(image_np, cmap='gray')
    axes[0, 0].set_title('原图 (绿色通道)')
    axes[0, 0].axis('off')
    
    # 2. ROI掩码
    axes[0, 1].imshow(roi_np, cmap='gray')
    axes[0, 1].set_title(f'ROI掩码 (覆盖率: {roi_np.mean():.1%})')
    axes[0, 1].axis('off')
    
    # 3. 金标血管
    axes[1, 0].imshow(vessel_np, cmap='gray')
    axes[1, 0].set_title(f'金标血管 (血管占比: {vessel_np.mean():.2%})')
    axes[1, 0].axis('off')
    
    # 4. 预测血管
    pred_display = pred_np * roi_np  # 应用ROI
    axes[1, 1].imshow(pred_display, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('预测血管 (概率图)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # 保存
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'验证集对比图已保存: {save_path} (样本ID: {img_id:02d})')
    plt.close()


def visualize_test_sample(
    model: torch.nn.Module,
    dataset: DRIVEDataset,
    device: str = 'cuda',
    sample_idx: Optional[int] = None,
    save_path: str = 'results/test_sample_prediction.png'
) -> None:
    """可视化测试集单个样本的预测结果。
    
    显示3张图：原图、ROI掩码、预测（测试集无金标）
    
    Args:
        model: 已加载的模型
        dataset: 测试集数据集
        device: 计算设备
        sample_idx: 样本索引（None则随机选择）
        save_path: 保存路径
    """
    model.eval()
    
    # 随机选择样本
    if sample_idx is None:
        sample_idx = random.randint(0, len(dataset) - 1)
    
    img_id = dataset.image_ids[sample_idx]
    image, vessel, roi = dataset[sample_idx]
    
    # 预测
    with torch.no_grad():
        image_batch = image.unsqueeze(0).to(device)
        output = model(image_batch)
        pred = torch.sigmoid(output[0, 0]).cpu()
    
    # 转为numpy
    image_np = tensor_to_numpy(image)
    roi_np = tensor_to_numpy(roi)
    pred_np = tensor_to_numpy(pred)
    
    # 应用ROI
    pred_roi = pred_np * roi_np
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'测试集样本预测结果 (ID: {img_id:02d})', fontsize=14)
    
    # 1. 原图
    axes[0].imshow(image_np, cmap='gray')
    axes[0].set_title('原图 (绿色通道)')
    axes[0].axis('off')
    
    # 2. ROI掩码
    axes[1].imshow(roi_np, cmap='gray')
    axes[1].set_title(f'ROI掩码 (覆盖率: {roi_np.mean():.1%})')
    axes[1].axis('off')
    
    # 3. 预测血管
    axes[2].imshow(pred_roi, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('预测血管 (概率图)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # 保存
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'测试集预测图已保存: {save_path} (样本ID: {img_id:02d})')
    plt.close()


def plot_training_curves(
    log_file: Path = Path('logs/training_log.csv'),
    save_path: Path = Path('results/training_curves.png')
) -> None:
    """生成训练曲线图（2×3布局，6个指标）。
    
    读取CSV日志文件并生成训练曲线，包含：
    - Dice系数（Train/Val）
    - 损失函数（Train）
    - IoU（Val）
    - CL-Break（Val）
    - Precision（Val）
    - Δβ₀（Val）
    
    Args:
        log_file: CSV日志文件路径
        save_path: 保存图像的路径
    """
    import pandas as pd
    
    if not log_file.exists():
        print(f'错误: 日志文件不存在: {log_file}')
        print('请先运行训练脚本生成日志文件。')
        return
    
    # 读取日志
    df = pd.read_csv(log_file)
    
    # 列名自适应：处理不同版本代码的列名差异
    # 旧版本用'cl_break', 'delta_beta0'，某些版本用'val_cl_break', 'val_delta_beta0'
    if 'val_cl_break' in df.columns and 'cl_break' not in df.columns:
        df = df.rename(columns={'val_cl_break': 'cl_break'})
    if 'val_delta_beta0' in df.columns and 'delta_beta0' not in df.columns:
        df = df.rename(columns={'val_delta_beta0': 'delta_beta0'})
    
    # 数据清洗：检测并处理多次训练运行的数据
    # 如果epoch有重复，只保留最后一次运行的数据
    if df['epoch'].duplicated().any():
        print(f'警告: 检测到多次训练运行的数据（{df["epoch"].duplicated().sum()} 个重复epoch）')
        # 方法：按epoch分组，保留每组最后一条（即最后一次运行的数据）
        df = df.drop_duplicates(subset=['epoch'], keep='last')
        print(f'已清洗：保留 {len(df)} 个唯一次的数据')
    
    # 按epoch排序确保曲线正确
    df = df.sort_values('epoch').reset_index(drop=True)
    
    # 创建2×3子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('训练曲线', fontsize=16)
    
    # 定义6个指标的配置
    metric_configs = [
        # 第一行
        {
            'ax': axes[0, 0],
            'title': 'Dice系数',
            'columns': ['train_dice', 'val_dice'],
            'labels': ['Train Dice', 'Val Dice'],
            'ylabel': 'Dice'
        },
        {
            'ax': axes[0, 1],
            'title': '损失函数',
            'columns': ['train_loss'],
            'labels': ['Train Loss'],
            'ylabel': 'Loss'
        },
        {
            'ax': axes[0, 2],
            'title': 'IoU',
            'columns': ['val_iou'],
            'labels': ['Val IoU'],
            'ylabel': 'IoU'
        },
        # 第二行
        {
            'ax': axes[1, 0],
            'title': 'CL-Break（越低越好）',
            'columns': ['cl_break'],  # 修正：与日志文件列名一致
            'labels': ['CL-Break'],
            'ylabel': 'CL-Break'
        },
        {
            'ax': axes[1, 1],
            'title': 'Precision',
            'columns': ['val_precision'],
            'labels': ['Val Precision'],
            'ylabel': 'Precision'
        },
        {
            'ax': axes[1, 2],
            'title': 'Δβ₀（Betti误差）',
            'columns': ['delta_beta0'],  # 修正：与日志文件列名一致
            'labels': ['Δβ₀'],
            'ylabel': 'Δβ₀'
        }
    ]
    
    # 绘制每个子图
    for config in metric_configs:
        ax = config['ax']
        
        for col, label in zip(config['columns'], config['labels']):
            if col in df.columns:
                ax.plot(df['epoch'], df[col], label=label, linewidth=1.5)
        
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(config['ylabel'], fontsize=10)
        ax.set_title(config['title'], fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'训练曲线已保存: {save_path}')
    plt.close()


def main(
    config_path: str = 'config.yaml',
    checkpoint_path: Optional[str] = None,
    val_sample_idx: Optional[int] = None,
    test_sample_idx: Optional[int] = None,
    plot_curves: bool = False
) -> None:
    """主函数。
    
    Args:
        config_path: 配置文件路径
        checkpoint_path: 模型检查点路径（None则使用best_model.pth）
        val_sample_idx: 验证集样本索引（None则随机选择）
        test_sample_idx: 测试集样本索引（None则随机选择）
        plot_curves: 是否仅生成训练曲线图
    """
    # 如果仅生成训练曲线
    if plot_curves:
        print('生成训练曲线图...')
        # 自动检测使用哪个日志文件
        log_file = Path('logs/training_topo_log.csv')
        if not log_file.exists():
            log_file = Path('logs/training_log.csv')
        save_path = Path('results/training_curves.png')
        plot_training_curves(log_file, save_path)
        return
    
    # 否则执行可视化流程
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设备
    device = config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    # 加载模型
    if checkpoint_path is None:
        checkpoint_path = config['training']['checkpoint_dir'] + '/best_model.pth'
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f'模型文件不存在: {checkpoint_path}\n请先运行训练脚本生成最佳模型。')
    
    print(f'加载模型: {checkpoint_path}')
    model = load_model(checkpoint_path, device)
    
    # 设置随机种子（确保随机样本可复现）
    random.seed(42)
    
    # 创建验证集数据集
    print('\n生成验证集可视化...')
    val_dataset = DRIVEDataset(
        root=config['data']['root'],
        image_ids=config['data']['val_ids'],
        img_size=config['data']['img_size'],
        in_channels=config['data']['in_channels'],
        is_training=False
    )
    
    visualize_val_sample(
        model, val_dataset, device,
        sample_idx=val_sample_idx,
        save_path='results/val_sample_comparison.png'
    )
    
    # 创建测试集数据集
    print('\n生成测试集可视化...')
    test_dataset = DRIVEDataset(
        root=config['data']['root'],
        image_ids=config['data']['test_ids'],
        img_size=config['data']['img_size'],
        in_channels=config['data']['in_channels'],
        is_training=False
    )
    
    visualize_test_sample(
        model, test_dataset, device,
        sample_idx=test_sample_idx,
        save_path='results/test_sample_prediction.png'
    )
    
    
    print('\n可视化完成！')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化模型预测结果')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型检查点路径')
    parser.add_argument('--val-idx', type=int, default=None, help='验证集样本索引（默认随机）')
    parser.add_argument('--test-idx', type=int, default=None, help='测试集样本索引（默认随机）')
    parser.add_argument('--plot-curves', action='store_true', help='仅生成训练曲线图')
    
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.val_idx, args.test_idx, args.plot_curves)
