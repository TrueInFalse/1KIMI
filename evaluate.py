# -*- coding: utf-8 -*-
"""
文件名: evaluate.py
项目: retina_ph_seg
功能: 模型评估脚本，生成可视化结果
作者: AI Assistant
创建日期: 2026-02-07
Python版本: 3.12+

依赖关系:
  - 上游: config.yaml, data_drive.py, model_unet.py, utils_metrics.py
  - 下游: 无（最终评估脚本）

主要类/函数:
  - evaluate_model(): 评估函数
  - visualize_results(): 可视化结果
  - main(): 入口函数

更新记录:
  - v1.0: 初始版本，实现评估和可视化
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_drive import DRIVEDataset, get_drive_loaders
from model_unet import load_model
from utils_metrics import compute_all_metrics, tensor_to_numpy


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    compute_topology: bool = True,
    save_predictions: bool = False,
    output_dir: Optional[Path] = None
) -> Dict[str, float]:
    """评估模型性能。
    
    Args:
        model: 已加载的模型
        dataloader: 数据加载器
        device: 计算设备
        compute_topology: 是否计算拓扑指标
        save_predictions: 是否保存预测结果
        output_dir: 预测结果保存目录
        
    Returns:
        metrics: 平均指标字典
    """
    model.eval()
    
    # 累积指标
    metrics_sum = {
        'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0,
        'cl_break': 0.0, 'delta_beta0': 0.0
    }
    num_samples = 0
    
    if save_predictions and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        pred_dir = output_dir / 'predictions'
        pred_dir.mkdir(exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (images, vessels, rois) in enumerate(tqdm(dataloader, desc='评估')):
            images = images.to(device)
            vessels = vessels.to(device)
            rois = rois.to(device)
            
            # 前向传播
            outputs = model(images)
            
            batch_size = images.shape[0]
            for i in range(batch_size):
                pred = torch.sigmoid(outputs[i, 0])
                target = vessels[i, 0]
                roi = rois[i, 0]
                
                # 转为numpy
                pred_np = tensor_to_numpy(pred)
                target_np = tensor_to_numpy(target)
                roi_np = tensor_to_numpy(roi)
                
                # 计算指标（ROI内）
                result = compute_all_metrics(
                    pred_np, target_np, roi_np,
                    threshold=0.5,
                    compute_topology=compute_topology
                )
                
                metrics_sum['dice'] += result.dice
                metrics_sum['iou'] += result.iou
                metrics_sum['precision'] += result.precision
                metrics_sum['recall'] += result.recall
                metrics_sum['cl_break'] += result.cl_break
                metrics_sum['delta_beta0'] += result.delta_beta0
                
                # 保存预测（可选）
                if save_predictions and output_dir:
                    img_id = dataloader.dataset.image_ids[num_samples]
                    np.save(
                        pred_dir / f'{img_id:02d}_pred.npy',
                        pred_np.astype(np.float32)
                    )
                
                num_samples += 1
    
    # 计算平均
    metrics = {k: v / num_samples for k, v in metrics_sum.items()}
    return metrics


def visualize_results(
    model: torch.nn.Module,
    dataset: DRIVEDataset,
    device: str = 'cuda',
    num_samples: int = 4,
    save_path: Optional[str] = None
) -> None:
    """可视化分割结果。
    
    每行显示：原图、GT血管、预测血管、GT骨架、预测骨架
    
    Args:
        model: 已加载的模型
        dataset: 数据集
        device: 计算设备
        num_samples: 可视化样本数
        save_path: 保存路径
    """
    from skimage.morphology import skeletonize
    
    model.eval()
    num_samples = min(num_samples, len(dataset))
    
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('分割结果可视化（从左到右：原图、GT血管、预测血管、GT骨架、预测骨架）', 
                 fontsize=12)
    
    with torch.no_grad():
        for idx in range(num_samples):
            # 获取样本
            img_id = dataset.image_ids[idx]
            image, vessel, roi = dataset[idx]
            
            # 预测
            image_batch = image.unsqueeze(0).to(device)
            output = model(image_batch)
            pred = torch.sigmoid(output[0, 0]).cpu().numpy()
            
            # 转为numpy
            image_np = tensor_to_numpy(image)
            vessel_np = tensor_to_numpy(vessel)
            roi_np = tensor_to_numpy(roi)
            
            # 应用ROI
            pred_roi = pred * roi_np
            vessel_roi = vessel_np * roi_np
            
            # 骨架化
            pred_binary = (pred_roi > 0.5).astype(np.uint8)
            vessel_binary = (vessel_roi > 0.5).astype(np.uint8)
            
            pred_skel = skeletonize(pred_binary) if pred_binary.sum() > 0 else np.zeros_like(pred_binary)
            vessel_skel = skeletonize(vessel_binary) if vessel_binary.sum() > 0 else np.zeros_like(vessel_binary)
            
            # 绘制
            # 1. 原图
            axes[idx, 0].imshow(image_np, cmap='gray')
            axes[idx, 0].set_title(f'ID:{img_id:02d} 原图')
            axes[idx, 0].axis('off')
            
            # 2. GT血管
            axes[idx, 1].imshow(vessel_roi, cmap='gray')
            axes[idx, 1].set_title('GT血管')
            axes[idx, 1].axis('off')
            
            # 3. 预测血管
            axes[idx, 2].imshow(pred_roi, cmap='gray', vmin=0, vmax=1)
            axes[idx, 2].set_title('预测血管')
            axes[idx, 2].axis('off')
            
            # 4. GT骨架
            axes[idx, 3].imshow(vessel_skel, cmap='gray')
            axes[idx, 3].set_title('GT骨架')
            axes[idx, 3].axis('off')
            
            # 5. 预测骨架
            axes[idx, 4].imshow(pred_skel, cmap='gray')
            axes[idx, 4].set_title('预测骨架')
            axes[idx, 4].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'可视化结果已保存: {save_path}')
    else:
        plt.savefig('results/evaluation_results.png', dpi=150, bbox_inches='tight')
        print('可视化结果已保存: results/evaluation_results.png')


def main(
    config_path: str = 'config.yaml',
    checkpoint_path: Optional[str] = None,
    split: str = 'val'
) -> None:
    """主函数。
    
    Args:
        config_path: 配置文件路径
        checkpoint_path: 模型检查点路径（None则使用best_model.pth）
        split: 评估数据集（'train'/'val'/'test'）
    """
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
        raise FileNotFoundError(f'模型文件不存在: {checkpoint_path}')
    
    print(f'加载模型: {checkpoint_path}')
    model = load_model(checkpoint_path, device)
    
    # 加载数据
    print(f'加载{split}数据集...')
    train_loader, val_loader, test_loader = get_drive_loaders(config_path)
    
    if split == 'train':
        loader = train_loader
    elif split == 'val':
        loader = val_loader
    else:
        loader = test_loader
    
    # 评估
    print('\n开始评估...')
    compute_topology = config.get('metrics', {}).get('compute_topology', True)
    
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    metrics = evaluate_model(
        model, loader, device,
        compute_topology=compute_topology,
        save_predictions=True,
        output_dir=results_dir
    )
    
    # 打印结果
    print('\n' + '=' * 60)
    print(f'评估结果 ({split}集)')
    print('=' * 60)
    print(f'Dice:        {metrics["dice"]:.4f}')
    print(f'IoU:         {metrics["iou"]:.4f}')
    print(f'Precision:   {metrics["precision"]:.4f}')
    print(f'Recall:      {metrics["recall"]:.4f}')
    if compute_topology:
        print(f'CL-Break:    {metrics["cl_break"]:.1f}')
        print(f'Δβ₀:         {metrics["delta_beta0"]:.1f}')
    print('=' * 60)
    
    # 可视化（仅对val集）
    if split in ['train', 'val']:
        dataset = loader.dataset
        visualize_results(
            model, dataset, device,
            num_samples=min(4, len(dataset)),
            save_path=str(results_dir / 'evaluation_results.png')
        )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='评估U-Net模型')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型检查点路径')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='评估数据集')
    
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.split)
