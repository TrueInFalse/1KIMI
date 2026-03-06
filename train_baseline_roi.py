# -*- coding: utf-8 -*-
"""
文件名: train_baseline_roi.py
项目: retina_ph_seg
功能: U-Net基线训练脚本（ROI对齐版 - Dice loss在ROI内计算）

与原版区别:
- 训练时Dice loss在ROI内计算（与验证口径一致）
- 其他逻辑保持不变

作者: AI Assistant
创建日期: 2026-03-06
"""

import os
import sys
import csv
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import segmentation_models_pytorch as smp

from data_combined import get_combined_loaders
from model_unet import get_unet_model, count_parameters
from utils_metrics import (
    compute_basic_metrics, 
    compute_topology_metrics,
    tensor_to_numpy
)


def compute_dice_loss_roi(pred_logits, target, roi_mask, eps=1e-7):
    """
    在ROI内计算Dice loss
    
    Args:
        pred_logits: [B, 1, H, W] 模型输出（logits）
        target: [B, 1, H, W] 目标标签
        roi_mask: [B, 1, H, W] ROI掩码
        eps: 平滑项
    
    Returns:
        loss: 标量
    """
    pred = torch.sigmoid(pred_logits)
    
    # 应用ROI mask
    pred = pred * roi_mask
    target = target * roi_mask
    
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    
    dice = (2.0 * intersection + eps) / (union + eps)
    loss = 1.0 - dice.mean()
    
    return loss


class EarlyStopping:
    """早停机制（仅监控val_dice）。"""
    
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-4,
        mode: str = 'max'
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, value: float, epoch: int) -> bool:
        if self.mode == 'max':
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class Trainer:
    """U-Net训练器（ROI对齐版）。"""
    
    def __init__(
        self,
        config: Dict,
        device: str = 'cuda'
    ) -> None:
        self.config = config
        self.device = device
        
        # 创建输出目录
        self.checkpoint_dir = Path(config['training']['checkpoint_dir'])
        self.log_dir = Path(config['training']['log_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化模型、优化器
        self._setup_model()
        self._setup_optimizer()
        
        # 早停机制
        self.enable_early_stopping = config['training'].get('enable_early_stopping', True)
        if self.enable_early_stopping:
            self.early_stopping = EarlyStopping(
                patience=config['training']['patience'],
                mode='max'
            )
        else:
            self.early_stopping = None
        
        # 日志（ROI对齐版）
        self.log_file = self.log_dir / 'training_baseline_roi_log.csv'
        self._init_log()
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_dice = 0.0
        self.global_step = 0
        
        # 时间记录
        self.start_time = None
        self.epoch_times = []
    
    def _setup_model(self) -> None:
        """初始化模型（统一RGB 3通道）。"""
        model_cfg = self.config['model']
        
        print('注意: 统一使用RGB 3通道输入（适配ImageNet预训练）')
        
        self.model = get_unet_model(
            in_channels=3,
            encoder=model_cfg['encoder'],
            pretrained=model_cfg['pretrained'],
            activation=model_cfg.get('activation', None)
        ).to(self.device)
        
        print(f'模型参数量: {count_parameters(self.model):,}')
    
    def _setup_optimizer(self) -> None:
        """初始化优化器和学习率调度器。"""
        train_cfg = self.config['training']
        
        # 优化器
        self.optimizer = Adam(
            self.model.parameters(),
            lr=train_cfg['learning_rate']
        )
        
        # 学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=train_cfg['max_epochs'],
            eta_min=train_cfg['learning_rate'] * 0.01
        )
    
    def _init_log(self) -> None:
        """初始化CSV日志文件（ROI对齐版）。"""
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_dice_loss', 'train_dice', 'val_dice',
                'val_iou', 'val_precision', 'val_recall',
                'val_cl_break', 'val_delta_beta0', 'lr', 'epoch_time',
                'roi_mode', 'roi_mean'
            ])
    
    def _log_epoch(
        self,
        epoch: int,
        train_dice_loss: float,
        train_dice: float,
        val_metrics: Dict[str, float],
        lr: float,
        roi_mode: str,
        roi_mean: float
    ) -> None:
        """记录一个epoch的结果。"""
        epoch_time = self.epoch_times[-1] if self.epoch_times else 0.0
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f'{train_dice_loss:.6f}',
                f'{train_dice:.4f}',
                f"{val_metrics.get('dice', 0):.4f}",
                f"{val_metrics.get('iou', 0):.4f}",
                f"{val_metrics.get('precision', 0):.4f}",
                f"{val_metrics.get('recall', 0):.4f}",
                f"{val_metrics.get('cl_break', 0):.1f}",
                f"{val_metrics.get('delta_beta0', 0):.1f}",
                f'{lr:.6f}',
                f'{epoch_time:.2f}',
                roi_mode,
                f'{roi_mean:.4f}'
            ])
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, float]:
        """训练一个epoch（ROI对齐版）。
        
        Returns:
            avg_loss: 平均Dice loss（ROI内）
            avg_dice: 平均Dice（简单估算）
            avg_roi_mean: 平均ROI覆盖度
        """
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        total_roi_mean = 0.0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch}', leave=False)
        
        for batch_idx, (images, vessels, rois) in enumerate(pbar):
            images = images.to(self.device)
            vessels = vessels.to(self.device)
            rois = rois.to(self.device)
            
            # 统计ROI
            with torch.no_grad():
                roi_mean_batch = rois.mean().item()
                total_roi_mean += roi_mean_batch
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # 计算损失（ROI内Dice loss）
            loss = compute_dice_loss_roi(outputs, vessels, rois)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            
            # 估算Dice（用于进度条显示）
            with torch.no_grad():
                pred = torch.sigmoid(outputs)
                pred_binary = (pred > 0.5).float()
                intersection = (pred_binary * vessels * rois).sum()
                dice = (2 * intersection) / ((pred_binary * rois).sum() + (vessels * rois).sum() + 1e-7)
                total_dice += dice.item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice.item():.4f}'
            })
            
            self.global_step += 1
        
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        avg_roi_mean = total_roi_mean / num_batches
        
        return avg_loss, avg_dice, avg_roi_mean
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证一个epoch。"""
        self.model.eval()
        
        metrics_sum = {
            'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0,
            'cl_break': 0.0, 'delta_beta0': 0.0
        }
        num_samples = 0
        
        compute_topology = self.config.get('metrics', {}).get('compute_topology', True)
        threshold = self.config.get('metrics', {}).get('topology_threshold', 0.5)
        
        with torch.no_grad():
            for images, vessels, rois in tqdm(val_loader, desc='验证', leave=False):
                images = images.to(self.device)
                vessels = vessels.to(self.device)
                rois = rois.to(self.device)
                
                outputs = self.model(images)
                
                batch_size = images.shape[0]
                for i in range(batch_size):
                    pred = torch.sigmoid(outputs[i, 0])
                    target = vessels[i, 0]
                    roi = rois[i, 0]
                    
                    pred_np = tensor_to_numpy(pred)
                    target_np = tensor_to_numpy(target)
                    roi_np = tensor_to_numpy(roi)
                    
                    basic = compute_basic_metrics(pred_np, target_np, roi_np, threshold)
                    
                    for key in ['dice', 'iou', 'precision', 'recall']:
                        metrics_sum[key] += basic[key]
                    
                    if compute_topology:
                        topo = compute_topology_metrics(pred_np, target_np, roi_np, threshold)
                        metrics_sum['cl_break'] += topo['cl_break']
                        metrics_sum['delta_beta0'] += topo['delta_beta0']
                    
                    num_samples += 1
        
        metrics = {k: v / num_samples for k, v in metrics_sum.items()}
        return metrics
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """保存模型检查点。"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_dice': self.best_val_dice,
            'config': {
                'in_channels': 3,
                'encoder': self.config['model']['encoder'],
                'img_size': self.config['data']['img_size']
            }
        }
        
        if is_best:
            path = self.checkpoint_dir / 'best_model_baseline_roi.pth'
            torch.save(checkpoint, path)
            print(f'  → 保存最佳模型: val_dice={self.best_val_dice:.4f}')
    
    def format_time(self, seconds: float) -> str:
        """格式化时间显示。"""
        td = timedelta(seconds=int(seconds))
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        secs = td.seconds % 60
        if hours > 0:
            return f'{hours}h {minutes}m {secs}s'
        elif minutes > 0:
            return f'{minutes}m {secs}s'
        else:
            return f'{secs}s'
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """完整训练流程。"""
        max_epochs = self.config['training']['max_epochs']
        
        self.start_time = time.time()
        start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print('\n' + '=' * 80)
        print('开始训练 (Baseline-ROI对齐版)')
        print(f'开始时间: {start_datetime}')
        print(f'设备: {self.device}')
        print(f'最大epoch: {max_epochs}')
        print('关键修改: Dice loss在ROI内计算')
        if self.enable_early_stopping:
            print(f'早停: 启用 (耐心值={self.config["training"]["patience"]})')
        else:
            print(f'早停: 禁用 (将跑满{max_epochs}轮)')
        print('=' * 80 + '\n')
        
        roi_mode = self.config.get('data', {}).get('kaggle_roi', {}).get('mode', 'fov')
        
        for epoch in range(1, max_epochs + 1):
            epoch_start_time = time.time()
            self.current_epoch = epoch
            
            train_dice_loss, train_dice, train_roi_mean = self.train_epoch(train_loader)
            
            val_metrics = self.validate(val_loader)
            val_dice = val_metrics['dice']
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            
            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)
            
            total_elapsed = time.time() - self.start_time
            avg_epoch_time = np.mean(self.epoch_times[-5:])
            remaining_epochs = max_epochs - epoch
            eta_seconds = avg_epoch_time * remaining_epochs
            
            print(f'\nEpoch {epoch}/{max_epochs}')
            print(f'  Train Dice Loss (ROI): {train_dice_loss:.4f} | Train Dice (ROI): {train_dice:.4f}')
            print(f'  Val Dice: {val_dice:.4f} | Val IoU: {val_metrics["iou"]:.4f} | Val Prec: {val_metrics["precision"]:.4f} | Val Rec: {val_metrics["recall"]:.4f}')
            print(f'  CL-Break: {val_metrics["cl_break"]:.1f} | Δβ₀: {val_metrics["delta_beta0"]:.1f}')
            print(f'  ROI Mean: {train_roi_mean:.4f} | ROI Mode: {roi_mode}')
            print(f'  Each Time: {self.format_time(epoch_time)} | Total Time: {self.format_time(total_elapsed)} | ETA: {self.format_time(eta_seconds)} | LR: {current_lr:.6f}')
            
            self._log_epoch(epoch, train_dice_loss, train_dice, val_metrics, current_lr, roi_mode, train_roi_mean)
            
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                self.save_checkpoint(is_best=True)
            
            if self.enable_early_stopping and self.early_stopping(val_dice, epoch):
                print(f'\n早停触发！最佳epoch: {self.early_stopping.best_epoch}, 最佳val_dice: {self.early_stopping.best_value:.4f}')
                break
        
        total_time = time.time() - self.start_time
        end_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print('\n' + '=' * 80)
        print('训练完成')
        print(f'开始时间: {start_datetime}')
        print(f'结束时间: {end_datetime}')
        print(f'总用时: {self.format_time(total_time)}')
        print(f'平均每轮: {self.format_time(np.mean(self.epoch_times))}')
        best_epoch_str = str(self.early_stopping.best_epoch) if self.early_stopping else 'N/A'
        print(f'最佳验证Dice: {self.best_val_dice:.4f} (Epoch {best_epoch_str})')
        print(f'最佳模型: {self.checkpoint_dir / "best_model_baseline_roi.pth"}')
        print('=' * 80)


def set_seed(seed: int = 42) -> None:
    """设置随机种子。"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(config_path: str = 'config.yaml') -> None:
    """主函数。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    set_seed(config['training'].get('seed', 42))
    
    device = config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA不可用，切换到CPU')
        device = 'cpu'
    
    print(f"当前数据模式: {'Kaggle联合' if config['data']['use_kaggle_combined'] else '纯DRIVE'}")
    print('加载数据...')
    train_loader, val_loader, _ = get_combined_loaders(config)
    
    trainer = Trainer(config, device)
    trainer.train(train_loader, val_loader)
    
    print('\n训练完成！')


if __name__ == '__main__':
    main('config.yaml')
