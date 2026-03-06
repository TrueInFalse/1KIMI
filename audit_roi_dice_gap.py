# -*- coding: utf-8 -*-
"""
文件名: audit_roi_dice_gap.py
项目: retina_ph_seg
功能: ROI Dice 口径审计脚本

审计目标:
    验证"训练目标与评估目标在ROI口径上错位"的问题，并量化错位程度。

审计逻辑:
    1. Baseline/Topo 训练时 Dice loss 计算在全图上（不使用 ROI）
    2. Topo loss 计算在 ROI 内
    3. 验证阶段所有指标（Dice/IoU/Precision/Recall）都在 ROI 内计算
    4. 这导致：训练优化的目标 ≠ 评估的目标

本脚本:
    - 对同一 batch 同时计算全图 Dice 和 ROI 内 Dice
    - 量化两者差距
    - 为"Baseline vs Topo 公平比较"提供数据支持
"""

import os
import sys
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

# 导入项目模块
from data_combined import get_combined_loaders
from model_unet import get_unet_model


def compute_dice_loss(pred_logits, target, roi_mask=None, eps=1e-7):
    """
    计算 Dice loss
    
    Args:
        pred_logits: [B, 1, H, W] 模型输出（logits）
        target: [B, 1, H, W] 目标标签
        roi_mask: [B, 1, H, W] ROI掩码（None表示全图）
        eps: 平滑项
    
    Returns:
        loss: 标量
    """
    pred = torch.sigmoid(pred_logits)
    
    if roi_mask is not None:
        # ROI 内计算
        pred = pred * roi_mask
        target = target * roi_mask
    
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    
    dice = (2.0 * intersection + eps) / (union + eps)
    loss = 1.0 - dice.mean()
    
    return loss


def compute_dice_score(pred_binary, target_binary, roi_mask=None, eps=1e-7):
    """
    计算 Dice 分数（用于二值化后的评估）
    
    Args:
        pred_binary: [B, 1, H, W] 二值预测
        target_binary: [B, 1, H, W] 二值标签
        roi_mask: [B, 1, H, W] ROI掩码（None表示全图）
        eps: 平滑项
    
    Returns:
        dice_scores: [B] 每个样本的Dice分数
    """
    if roi_mask is not None:
        pred_binary = pred_binary * roi_mask
        target_binary = target_binary * roi_mask
    
    intersection = (pred_binary * target_binary).sum(dim=(1, 2, 3))
    union = pred_binary.sum(dim=(1, 2, 3)) + target_binary.sum(dim=(1, 2, 3))
    
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice


def normalize_roi_tensor(rois, device):
    """将ROI批次统一为 [B, 1, H, W]。"""
    if isinstance(rois, torch.Tensor):
        roi_tensor = rois.to(device)
    elif isinstance(rois, (list, tuple)):
        roi_tensor = torch.stack([r.to(device) for r in rois])
    else:
        raise TypeError(f'不支持的ROI类型: {type(rois)}')

    if roi_tensor.dim() == 3:
        roi_tensor = roi_tensor.unsqueeze(1)
    elif roi_tensor.dim() == 4:
        pass
    else:
        raise ValueError(f'ROI张量维度异常: {roi_tensor.shape}')

    return roi_tensor.float()


def save_overlay_figures(image, roi_mask, pred_prob, batch_idx, sample_idx, save_dir, diff_value):
    """保存可视化叠加图"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 转换为numpy
    img_np = image.cpu().numpy()  # [3, H, W]
    roi_np = roi_mask[0].cpu().numpy()  # [1, H, W] -> [H, W]
    pred_np = pred_prob[0].cpu().numpy()  # [1, H, W] -> [H, W]
    
    # 反归一化图像（假设mean=0.5, std=0.5）
    img_np = img_np * 0.5 + 0.5
    img_np = np.clip(img_np, 0, 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原图
    axes[0].imshow(img_np.transpose(1, 2, 0))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # ROI mask
    axes[1].imshow(roi_np, cmap='gray')
    axes[1].set_title(f'ROI Mask (mean={roi_np.mean():.3f})')
    axes[1].axis('off')
    
    # 预测概率 + ROI叠加
    axes[2].imshow(pred_np, cmap='hot', vmin=0, vmax=1)
    axes[2].contour(roi_np, colors='cyan', linewidths=1)
    axes[2].set_title(f'Pred Prob + ROI (diff={diff_value:.4f})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / f'batch{batch_idx}_sample{sample_idx}_diff{diff_value:.4f}.png', dpi=150)
    plt.close()


def audit_roi_dice_gap(config_path='config.yaml', max_batches=None):
    """
    主审计函数
    
    Args:
        config_path: 配置文件路径
        max_batches: 最大审计batch数（None表示全部）
    """
    print("=" * 80)
    print("ROI Dice 口径审计")
    print("=" * 80)
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print(f"数据模式: {'Kaggle联合' if config['data']['use_kaggle_combined'] else '纯DRIVE'}")
    print(f"ROI模式: {config['data'].get('kaggle_roi', {}).get('mode', 'N/A')}")
    
    # 加载数据
    print("\n加载数据...")
    train_loader, val_loader, _ = get_combined_loaders(config)
    
    # 初始化模型（随机权重即可，因为我们只关心输入输出的数值关系）
    print("初始化模型（随机权重）...")
    # 设置随机种子确保可复现
    torch.manual_seed(42)
    model = get_unet_model(
        in_channels=3,
        encoder=config['model']['encoder'],
        pretrained=False,  # 不需要预训练权重
        activation=config['model'].get('activation', None)
    ).to(device)
    model.eval()
    # 关闭BN的随机性
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.Dropout)):
            m.eval()
    
    # 审计统计
    stats = {
        'train': {'full_dice': [], 'roi_dice': [], 'diff': [], 'roi_mean': []},
        'val': {'full_dice': [], 'roi_dice': [], 'diff': [], 'roi_mean': []}
    }
    
    # 记录最大差异的样本
    max_diff_samples = []  # [(split, batch_idx, sample_idx, diff, full_dice, roi_dice), ...]
    
    # 用于辅助统计的一个batch详细数据
    detailed_batch_info = None
    
    print("\n" + "=" * 80)
    print("开始审计 Train 集...")
    print("=" * 80)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Train Audit")):
            if max_batches and batch_idx >= max_batches:
                break
            
            # 解析batch
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                images, vessels, rois = batch
                images = images.to(device)
                vessels = vessels.to(device)
                rois = normalize_roi_tensor(rois, device)
            else:
                images = batch['image'].to(device)
                vessels = batch['vessel'].to(device)
                rois = normalize_roi_tensor(batch['roi'], device)
            
            # 前向传播
            outputs = model(images)
            pred_prob = torch.sigmoid(outputs)
            pred_binary = (pred_prob > 0.5).float()
            vessels_binary = (vessels > 0.5).float()
            
            # 计算全图 Dice loss
            full_dice_loss = compute_dice_loss(outputs, vessels, roi_mask=None)
            
            # 计算 ROI 内 Dice loss
            roi_dice_loss = compute_dice_loss(outputs, vessels, roi_mask=rois)
            
            # 计算全图 Dice score（二值化后）
            full_dice_scores = compute_dice_score(pred_binary, vessels_binary, roi_mask=None)
            
            # 计算 ROI 内 Dice score（二值化后）
            roi_dice_scores = compute_dice_score(pred_binary, vessels_binary, roi_mask=rois)
            
            # 计算差异
            diff_scores = torch.abs(full_dice_scores - roi_dice_scores)
            
            # 记录统计
            for i in range(len(images)):
                full_dice = full_dice_scores[i].item()
                roi_dice = roi_dice_scores[i].item()
                diff = diff_scores[i].item()
                roi_mean = rois[i].mean().item()
                
                stats['train']['full_dice'].append(full_dice)
                stats['train']['roi_dice'].append(roi_dice)
                stats['train']['diff'].append(diff)
                stats['train']['roi_mean'].append(roi_mean)
                
                # 记录最大差异样本
                max_diff_samples.append(('train', batch_idx, i, diff, full_dice, roi_dice, roi_mean))
            
            # 保存第一个batch的详细信息用于辅助统计
            if batch_idx == 0 and detailed_batch_info is None:
                detailed_batch_info = {
                    'split': 'train',
                    'batch_idx': batch_idx,
                    'images': images.cpu(),
                    'rois': rois.cpu(),
                    'pred_prob': pred_prob.cpu(),
                    'vessels': vessels.cpu(),
                    'full_dice_scores': full_dice_scores.cpu(),
                    'roi_dice_scores': roi_dice_scores.cpu(),
                }
            
            # 保存差异大的可视化图（每10个batch保存一个）
            if batch_idx % 10 == 0:
                max_diff_idx = torch.argmax(diff_scores).item()
                if diff_scores[max_diff_idx].item() > 0.05:  # 差异>5%才保存
                    save_overlay_figures(
                        images[max_diff_idx], 
                        rois[max_diff_idx], 
                        pred_prob[max_diff_idx],
                        batch_idx, max_diff_idx,
                        'results/roi_audit',
                        diff_scores[max_diff_idx].item()
                    )
    
    print("\n" + "=" * 80)
    print("开始审计 Val 集...")
    print("=" * 80)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Val Audit")):
            if max_batches and batch_idx >= max_batches:
                break
            
            # 解析batch
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                images, vessels, rois = batch
                images = images.to(device)
                vessels = vessels.to(device)
                rois = normalize_roi_tensor(rois, device)
            else:
                images = batch['image'].to(device)
                vessels = batch['vessel'].to(device)
                rois = normalize_roi_tensor(batch['roi'], device)
            
            # 前向传播
            outputs = model(images)
            pred_prob = torch.sigmoid(outputs)
            pred_binary = (pred_prob > 0.5).float()
            vessels_binary = (vessels > 0.5).float()
            
            # 计算全图 Dice score（二值化后）
            full_dice_scores = compute_dice_score(pred_binary, vessels_binary, roi_mask=None)
            
            # 计算 ROI 内 Dice score（二值化后）
            roi_dice_scores = compute_dice_score(pred_binary, vessels_binary, roi_mask=rois)
            
            # 计算差异
            diff_scores = torch.abs(full_dice_scores - roi_dice_scores)
            
            # 记录统计
            for i in range(len(images)):
                full_dice = full_dice_scores[i].item()
                roi_dice = roi_dice_scores[i].item()
                diff = diff_scores[i].item()
                roi_mean = rois[i].mean().item()
                
                stats['val']['full_dice'].append(full_dice)
                stats['val']['roi_dice'].append(roi_dice)
                stats['val']['diff'].append(diff)
                stats['val']['roi_mean'].append(roi_mean)
                
                # 记录最大差异样本
                max_diff_samples.append(('val', batch_idx, i, diff, full_dice, roi_dice, roi_mean))
    
    # ==================== 输出统计结果 ====================
    print("\n" + "=" * 80)
    print("审计结果统计")
    print("=" * 80)
    
    for split in ['train', 'val']:
        if not stats[split]['full_dice']:
            continue
            
        print(f"\n【{split.upper()} 集】")
        print("-" * 60)
        
        full_dice_arr = np.array(stats[split]['full_dice'])
        roi_dice_arr = np.array(stats[split]['roi_dice'])
        diff_arr = np.array(stats[split]['diff'])
        roi_mean_arr = np.array(stats[split]['roi_mean'])
        
        # 基础统计
        print(f"样本数: {len(full_dice_arr)}")
        print(f"\n全图 Dice - Mean: {full_dice_arr.mean():.4f}, Std: {full_dice_arr.std():.4f}")
        print(f"         - Min: {full_dice_arr.min():.4f}, Max: {full_dice_arr.max():.4f}")
        print(f"\nROI  Dice - Mean: {roi_dice_arr.mean():.4f}, Std: {roi_dice_arr.std():.4f}")
        print(f"         - Min: {roi_dice_arr.min():.4f}, Max: {roi_dice_arr.max():.4f}")
        print(f"\n绝对差值 - Mean: {diff_arr.mean():.4f}, Std: {diff_arr.std():.4f}")
        print(f"         - Min: {diff_arr.min():.4f}, Max: {diff_arr.max():.4f}")
        print(f"         - Median: {np.median(diff_arr):.4f}")
        
        # ROI 覆盖度统计
        print(f"\nROI 覆盖度 - Min: {roi_mean_arr.min():.4f}, Max: {roi_mean_arr.max():.4f}")
        print(f"          - Mean: {roi_mean_arr.mean():.4f}, Median: {np.median(roi_mean_arr):.4f}")
        
        # 全1 ROI样本数
        roi_all_ones = np.sum(roi_mean_arr >= 0.999)
        print(f"全1 ROI样本数: {roi_all_ones} ({roi_all_ones/len(roi_mean_arr)*100:.1f}%)")
        
        # 差值分布
        print(f"\n差值分布:")
        thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
        for thresh in thresholds:
            count = np.sum(diff_arr > thresh)
            print(f"  > {thresh:.3f}: {count} ({count/len(diff_arr)*100:.1f}%)")
    
    # 差异最大的前5个样本
    print("\n" + "=" * 80)
    print("差异最大的前5个样本")
    print("=" * 80)
    max_diff_samples.sort(key=lambda x: x[3], reverse=True)
    for i, (split, batch_idx, sample_idx, diff, full_dice, roi_dice, roi_mean) in enumerate(max_diff_samples[:5]):
        print(f"{i+1}. [{split}] batch={batch_idx}, sample={sample_idx}")
        print(f"   差值={diff:.4f}, 全图Dice={full_dice:.4f}, ROI Dice={roi_dice:.4f}, ROI覆盖={roi_mean:.3f}")
    
    # 辅助统计：第一个batch的详细分析
    if detailed_batch_info is not None:
        print("\n" + "=" * 80)
        print("第一个 Train Batch 的详细分析")
        print("=" * 80)
        
        batch_size = detailed_batch_info['images'].shape[0]
        rois = detailed_batch_info['rois']  # [B, 1, H, W]
        pred_prob = detailed_batch_info['pred_prob']  # [B, 1, H, W]
        
        for i in range(min(3, batch_size)):  # 只看前3个样本
            roi = rois[i, 0]  # [H, W]
            pred = pred_prob[i, 0]  # [H, W]
            
            roi_mask_bool = (roi > 0.5)
            
            if roi_mask_bool.any():
                pred_in_roi = pred[roi_mask_bool].mean().item()
                pred_out_roi = pred[~roi_mask_bool].mean().item() if (~roi_mask_bool).any() else 0.0
            else:
                pred_in_roi = pred_out_roi = 0.0
            
            print(f"\n样本 {i}:")
            print(f"  ROI覆盖度: {roi.mean().item():.4f}")
            print(f"  Pred均值(ROI内): {pred_in_roi:.4f}")
            print(f"  Pred均值(ROI外): {pred_out_roi:.4f}")
            print(f"  Pred均值差(外-内): {pred_out_roi - pred_in_roi:.4f}")
    
    # 保存详细结果到文件
    result_dir = Path('results/roi_audit')
    result_dir.mkdir(parents=True, exist_ok=True)
    
    with open(result_dir / 'audit_results.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ROI Dice 口径审计结果\n")
        f.write("=" * 80 + "\n\n")
        
        for split in ['train', 'val']:
            if not stats[split]['full_dice']:
                continue
            
            f.write(f"【{split.upper()} 集】\n")
            f.write("-" * 60 + "\n")
            
            full_dice_arr = np.array(stats[split]['full_dice'])
            roi_dice_arr = np.array(stats[split]['roi_dice'])
            diff_arr = np.array(stats[split]['diff'])
            roi_mean_arr = np.array(stats[split]['roi_mean'])
            
            f.write(f"样本数: {len(full_dice_arr)}\n")
            f.write(f"全图 Dice Mean: {full_dice_arr.mean():.4f} ± {full_dice_arr.std():.4f}\n")
            f.write(f"ROI  Dice Mean: {roi_dice_arr.mean():.4f} ± {roi_dice_arr.std():.4f}\n")
            f.write(f"绝对差值 Mean: {diff_arr.mean():.4f} ± {diff_arr.std():.4f}\n")
            f.write(f"绝对差值 Median: {np.median(diff_arr):.4f}\n")
            f.write(f"绝对差值 Max: {diff_arr.max():.4f}\n")
            f.write(f"ROI覆盖度 Mean: {roi_mean_arr.mean():.4f}\n")
            f.write(f"\n")
    
    print(f"\n详细结果已保存到: {result_dir / 'audit_results.txt'}")
    
    return stats


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ROI Dice 口径审计')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--max-batches', type=int, default=None, help='最大审计batch数')
    args = parser.parse_args()
    
    audit_roi_dice_gap(args.config, args.max_batches)
