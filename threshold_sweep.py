#!/usr/bin/env python
"""
阈值扫描脚本 - 确定最优二值化阈值
策略B: 在验证集扫描，选定最优阈值后冻结
"""

import argparse
import numpy as np
import torch
import yaml
from pathlib import Path
from tqdm import tqdm

from train_with_topology import TrainerWithTopology
from data_combined import get_combined_loaders
from utils_metrics import compute_basic_metrics, compute_topology_metrics


def threshold_sweep(model, val_loader, device, thresholds=np.arange(0.3, 0.8, 0.05)):
    """
    在验证集上扫描不同阈值，记录指标变化
    
    Args:
        model: 训练好的模型
        val_loader: 验证集加载器
        device: 计算设备
        thresholds: 待扫描的阈值列表
        
    Returns:
        results: 字典 {threshold: {dice, cl_break, delta_beta0}}
    """
    model.eval()
    results = {th: {'dice': [], 'cl_break': [], 'delta_beta0': []} for th in thresholds}
    
    print(f"开始阈值扫描: {thresholds}")
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='阈值扫描'):
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                images, vessels, rois = batch
                images = images.to(device)
                vessels = vessels.to(device)
                rois = rois.to(device)
            else:
                images = batch['image'].to(device)
                vessels = batch['vessel'].to(device)
                rois = batch['roi'].to(device)
            
            # 前向传播
            outputs = model(images)
            pred_prob = torch.sigmoid(outputs)
            
            # 对每个阈值计算指标
            for threshold in thresholds:
                for i in range(images.shape[0]):
                    pred = (pred_prob[i, 0] > threshold).cpu().numpy()
                    target = vessels[i, 0].cpu().numpy()
                    roi = rois[i, 0].cpu().numpy() if rois.dim() > 2 else rois[i].cpu().numpy()
                    
                    # 基础指标
                    basic = compute_basic_metrics(pred, target, roi, threshold)
                    results[threshold]['dice'].append(basic['dice'])
                    
                    # 拓扑指标
                    topo = compute_topology_metrics(pred, target, roi, threshold)
                    results[threshold]['cl_break'].append(topo['cl_break'])
                    results[threshold]['delta_beta0'].append(topo['delta_beta0'])
    
    # 计算平均值
    summary = {}
    for th in thresholds:
        summary[th] = {
            'dice': np.mean(results[th]['dice']),
            'cl_break': np.mean(results[th]['cl_break']),
            'delta_beta0': np.mean(results[th]['delta_beta0'])
        }
    
    return summary


def find_optimal_threshold(summary, metric='dice'):
    """
    根据指定指标选择最优阈值
    
    Args:
        summary: 阈值扫描结果
        metric: 优化目标 ('dice', 'cl_break', 'combined')
        
    Returns:
        optimal_threshold: 最优阈值
    """
    if metric == 'dice':
        # 最大化Dice
        optimal = max(summary.items(), key=lambda x: x[1]['dice'])
    elif metric == 'cl_break':
        # 最小化CL-Break
        optimal = min(summary.items(), key=lambda x: x[1]['cl_break'])
    elif metric == 'combined':
        # 综合指标: Dice - 0.01*CL_Break
        scores = {th: v['dice'] - 0.01 * v['cl_break'] for th, v in summary.items()}
        optimal = max(scores.items(), key=lambda x: x[1])
        optimal = (optimal[0], summary[optimal[0]])
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return optimal[0], optimal[1]


def print_results(summary):
    """打印扫描结果表格"""
    print("\n" + "="*80)
    print("阈值扫描结果")
    print("="*80)
    print(f"{'Threshold':>10} | {'Dice':>10} | {'CL-Break':>10} | {'Δβ₀':>10}")
    print("-"*80)
    
    for th, metrics in sorted(summary.items()):
        print(f"{th:>10.2f} | {metrics['dice']:>10.4f} | {metrics['cl_break']:>10.2f} | {metrics['delta_beta0']:>10.2f}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='阈值扫描')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件')
    parser.add_argument('--output', type=str, default='threshold_sweep_results.yaml', help='输出文件')
    parser.add_argument('--metric', type=str, default='dice', choices=['dice', 'cl_break', 'combined'])
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 加载数据
    _, val_loader, _ = get_combined_loaders(config)
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 简化初始化，只加载模型权重
    from model_unet import get_unet_model
    model = get_unet_model(in_channels=3, encoder='resnet34', pretrained=False).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"加载模型: {args.checkpoint}")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Val Dice: {checkpoint.get('val_dice', 'unknown')}")
    
    # 阈值扫描
    thresholds = np.arange(0.3, 0.75, 0.05)
    summary = threshold_sweep(model, val_loader, device, thresholds)
    
    # 打印结果
    print_results(summary)
    
    # 选择最优阈值
    optimal_th, optimal_metrics = find_optimal_threshold(summary, args.metric)
    
    print(f"\n最优阈值 (基于{args.metric}): {optimal_th:.2f}")
    print(f"  Dice: {optimal_metrics['dice']:.4f}")
    print(f"  CL-Break: {optimal_metrics['cl_break']:.2f}")
    print(f"  Δβ₀: {optimal_metrics['delta_beta0']:.2f}")
    
    # 保存结果
    results = {
        'thresholds': {float(k): v for k, v in summary.items()},
        'optimal_threshold': float(optimal_th),
        'optimal_metrics': optimal_metrics,
        'selection_metric': args.metric,
        'checkpoint': args.checkpoint
    }
    
    with open(args.output, 'w') as f:
        yaml.dump(results, f)
    
    print(f"\n结果已保存: {args.output}")


if __name__ == '__main__':
    main()
