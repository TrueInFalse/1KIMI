# -*- coding: utf-8 -*-
"""
文件名: train_with_topology.py (cripser 0.0.25+ 版本)
项目: retina_ph_seg
功能: 端到端拓扑正则训练（使用真正的可微分持续同调）
作者: AI Assistant
创建日期: 2026-02-19
Python版本: 3.12+

依赖关系:
  - 上游: config.yaml, data_drive.py, model_unet.py, utils_metrics.py, topology_loss.py
  - 下游: evaluate.py

技术路线 (cripser 0.0.25+):
  - 使用cripser.compute_ph_torch实现真正的可微分持续同调
  - 不再需要STE（Straight-Through Estimator）
  - 梯度真实传播，拓扑模块真正生效

更新记录:
  - v2.0: 基于cripser 0.0.25重写，移除STE，使用真正的可微分持续同调
  - v1.0: STE架构（已废弃）
"""

import os
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import segmentation_models_pytorch as smp

from data_combined import get_combined_loaders
# from data_drive import get_drive_loaders  # 直接使用data_combined.py中的加载器，包含DRIVE数据集
from model_unet import get_unet_model, count_parameters
from utils_metrics import compute_basic_metrics, compute_topology_metrics, tensor_to_numpy
from topology_loss import CubicalRipserLoss


class LambdaScheduler:
    """
    λ课程学习调度器（修正版）
    
    策略：前30 epoch固定0.1，后70 epoch线性增至0.5
    """
    
    def __init__(
        self,
        warmup_epochs: int = 30,      # 前30轮固定
        ramp_epochs: int = 70,         # 后70轮线性增长
        lambda_start: float = 0.1,     # 起始λ
        lambda_end: float = 0.5        # 最终λ
    ):
        self.warmup = warmup_epochs
        self.ramp = ramp_epochs
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
    
    def get_lambda(self, epoch: int) -> float:
        """获取当前epoch的λ值。"""
        if epoch < self.warmup:
            return self.lambda_start
        elif epoch < self.warmup + self.ramp:
            # 线性插值
            progress = (epoch - self.warmup) / self.ramp
            return self.lambda_start + progress * (self.lambda_end - self.lambda_start)
        else:
            return self.lambda_end


class TrainerWithTopology:
    """带拓扑正则的训练器（cripser 0.0.25+ 真正可微分版）。
    
    使用cripser.compute_ph_torch实现真正的可微分持续同调：
    - 不再需要STE近似
    - 梯度真实流经拓扑模块
    - 拓扑损失真正影响模型参数
    
    总损失：
        L_total = L_Dice + λ * L_topo
    """
    
    def __init__(
        self,
        config: Dict,
        args: Optional[Any] = None
    ) -> None:
        self.config = config
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 输出目录
        self.checkpoint_dir = Path('./checkpoints')
        self.log_dir = Path('./logs')
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # 初始化模型
        self._setup_model()
        
        # 损失函数
        self.criterion_dice = smp.losses.DiceLoss(mode='binary', from_logits=True)
        # 拓扑损失（支持自适应target_beta0）
        topo_cfg = config.get('topology', {})
        self.criterion_topo = CubicalRipserLoss(
            target_beta0=topo_cfg.get('target_beta0'),  # None表示自动计算
            max_death=topo_cfg.get('max_death', 0.5),
            loss_scale=topo_cfg.get('loss_scale', 300.0),
            excess_weight=topo_cfg.get('excess_weight', 0.3),
            short_weight=topo_cfg.get('short_weight', 0.7),
            short_threshold=topo_cfg.get('short_threshold', 0.08)
        ).to(self.device)
        
        # λ调度器（修正版：前30固定0.1，后70线性增至0.5）
        self.lambda_scheduler = LambdaScheduler(
            warmup_epochs=30,
            ramp_epochs=70,
            lambda_start=0.1,
            lambda_end=0.5
        )
        
        # 日志
        self.log_file = self.log_dir / 'training_topo_log.csv'
        self._init_log(overwrite=True)  # 新训练时清空旧日志
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_dice = 0.0
        self.start_time = None
        
        print(f'设备: {self.device}')
        print(f'拓扑损失: CubicalRipserLoss (cripser 0.0.25+ 真正可微分)')
    
    def _setup_model(self) -> None:
        """初始化模型（统一RGB 3通道）。"""
        model_cfg = self.config['model']
        
        # 统一使用RGB 3通道（适配ImageNet预训练权重）
        self.in_channels = 3
        print('注意: 统一使用RGB 3通道输入（适配ImageNet预训练）')
        
        self.model = get_unet_model(
            in_channels=3,
            encoder=model_cfg['encoder'],
            pretrained=model_cfg['pretrained'],
            activation=model_cfg.get('activation', None)
        ).to(self.device)
        
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate']
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['max_epochs'],
            eta_min=self.config['training']['learning_rate'] * 0.01
        )
        
        print(f'模型参数量: {count_parameters(self.model):,}')
    
    def _init_log(self, overwrite: bool = False) -> None:
        """初始化日志文件。
        
        Args:
            overwrite: 是否覆盖已有文件（新训练时设为True）
        """
        if overwrite or not self.log_file.exists():
            with open(self.log_file, 'w') as f:
                f.write('epoch,train_loss,train_dice,train_loss_topo,'
                       'val_dice,val_iou,val_precision,val_recall,'
                       'cl_break,delta_beta0,lambda,lr\n')
            if overwrite and self.log_file.exists():
                print(f'注意: 已清空旧日志文件 {self.log_file}')
    
    def format_time(self, seconds: float) -> str:
        """格式化时间为可读字符串。"""
        if seconds < 60:
            return f'{seconds:.0f}s'
        elif seconds < 3600:
            return f'{seconds/60:.0f}m {seconds%60:.0f}s'
        else:
            return f'{seconds/3600:.0f}h {(seconds%3600)/60:.0f}m'
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, float]:
        """训练一个epoch。
        
        Returns:
            avg_loss: 平均总损失
            avg_dice: 平均Dice
            avg_loss_topo: 平均拓扑损失
        """
        self.model.train()
        
        total_loss = 0.0
        total_dice = 0.0
        total_loss_topo = 0.0
        num_batches = len(train_loader)
        
        # 获取当前λ
        current_lambda = self.lambda_scheduler.get_lambda(self.current_epoch)
        
        for batch_idx, batch in enumerate(train_loader):
            # 处理list格式的batch [image, vessel, roi]
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                images, vessels, rois = batch
                images = images.to(self.device)
                vessels = vessels.to(self.device)
                rois = [r.to(self.device) for r in rois]
            else:
                # dict格式（向后兼容）
                images = batch['image'].to(self.device)
                vessels = batch['vessel'].to(self.device)
                rois = [r.to(self.device) for r in batch['roi']]
            
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(images)
            
            # Dice损失（始终计算）
            loss_dice = self.criterion_dice(outputs, vessels)
            
            # 拓扑损失（cripser 0.0.25+ 真正可微分！）
            pred = torch.sigmoid(outputs)
            
            # 将rois列表转为tensor [B, 1, H, W]
            roi_tensor = torch.stack([r for r in rois]).unsqueeze(1).float()
            loss_topo = self.criterion_topo(pred, roi_tensor, self.current_epoch)
            
            # 总损失
            loss = loss_dice + current_lambda * loss_topo
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_loss_topo += loss_topo.item()
            
            with torch.no_grad():
                pred_binary = (pred > 0.5).float()
                dice = (2 * (pred_binary * vessels).sum()) / (pred_binary.sum() + vessels.sum() + 1e-7)
                total_dice += dice.item()
            
            self.global_step += 1
        
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        avg_loss_topo = total_loss_topo / num_batches
        
        return avg_loss, avg_dice, avg_loss_topo
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证。
        
        Returns:
            metrics: 包含val_dice, val_iou, cl_break等的字典
        """
        self.model.eval()
        
        all_preds = []
        all_masks = []
        all_rois = []
        
        for batch in tqdm(val_loader, desc='Validate'):
            # 处理list格式的batch [image, vessel, roi]
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                images, vessels, rois = batch
                images = images.to(self.device)
                vessels = vessels.to(self.device)
                rois = [r.to(self.device) for r in rois]
            else:
                images = batch['image'].to(self.device)
                vessels = batch['vessel'].to(self.device)
                rois = [r.to(self.device) for r in batch['roi']]
            
            outputs = self.model(images)
            pred = torch.sigmoid(outputs)
            
            all_preds.append(pred)
            all_masks.append(vessels)
            all_rois.extend(rois)
        
        # 拼接所有批次并转为numpy
        all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
        all_masks = torch.cat(all_masks, dim=0).cpu().numpy()
        all_rois = [r.cpu().numpy() for r in all_rois]
        
        # 逐个样本计算指标并取平均
        all_dice, all_iou, all_prec, all_rec = [], [], [], []
        all_cl_break, all_delta_beta0 = [], []
        
        for i in range(len(all_preds)):
            roi = all_rois[i]
            if roi.ndim == 3:
                roi = roi[0]  # squeeze channel dim
            
            # 基础指标
            m = compute_basic_metrics(
                all_preds[i, 0],  # [H, W]
                all_masks[i, 0],  # [H, W]
                roi               # [H, W]
            )
            all_dice.append(m['dice'])
            all_iou.append(m['iou'])
            all_prec.append(m['precision'])
            all_rec.append(m['recall'])
            
            # 拓扑指标
            try:
                topo_m = compute_topology_metrics(
                    all_preds[i, 0],
                    all_masks[i, 0],
                    roi
                )
                all_cl_break.append(topo_m['cl_break'])
                all_delta_beta0.append(topo_m['delta_beta0'])
            except:
                all_cl_break.append(0.0)
                all_delta_beta0.append(0.0)
        
        metrics = {
            'dice': np.mean(all_dice),
            'iou': np.mean(all_iou),
            'precision': np.mean(all_prec),
            'recall': np.mean(all_rec),
            'cl_break': np.mean(all_cl_break) if all_cl_break else 0.0,
            'delta_beta0': np.mean(all_delta_beta0) if all_delta_beta0 else 0.0
        }
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """主训练循环。"""
        self.start_time = datetime.now()
        max_epochs = self.config['training']['max_epochs']
        
        
        # 如果需要，从训练集计算target_beta0
        if hasattr(self.criterion_topo, '_auto_compute_beta0') and self.criterion_topo._auto_compute_beta0:
            print("\n[拓扑损失] 从训练集金标计算target_beta0...")
            target_beta0 = self.criterion_topo.compute_target_beta0_from_loader(train_loader)
            self.criterion_topo.target_beta0 = target_beta0
            self.criterion_topo._auto_compute_beta0 = False
            print(f"[拓扑损失] 设置target_beta0 = {target_beta0}")
        
        # 打印拓扑损失配置
        topo_stats = self.criterion_topo.get_stats()
        print(f"\n[拓扑损失配置]")
        print(f"  target_beta0: {topo_stats['target_beta0']}")
        print(f"  loss_scale: {topo_stats['loss_scale']}")
        print(f"  excess_weight: {topo_stats['excess_weight']}")
        print(f"  short_weight: {topo_stats['short_weight']}")
        print(f"  short_threshold: {topo_stats['short_threshold']}")
        print(f'\\n开始训练 (cripser 0.0.25+ 真正可微分版)')
        print(f'总轮数: {max_epochs}')
        print(f'开始时间: {self.start_time.strftime("%Y-%m-%d %H:%M:%S")}')
        print('=' * 80)
        
        for epoch in range(max_epochs):
            self.current_epoch = epoch + 1
            
            # 训练
            train_loss, train_dice, train_loss_topo = self.train_epoch(train_loader)
            
            # 验证
            val_metrics = self.validate(val_loader)
            val_dice = val_metrics['dice']
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # 获取当前λ
            current_lambda = self.lambda_scheduler.get_lambda(epoch)
            
            # 计算时间
            elapsed = (datetime.now() - self.start_time).total_seconds()
            epoch_time = elapsed / self.current_epoch
            eta = epoch_time * (max_epochs - self.current_epoch)
            
            # 打印日志（美观格式，类似train_baseline.py）
            print(f'\nEpoch {self.current_epoch}/{max_epochs}  (λ={current_lambda:.3f})')
            print(f'  Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | Train Topo: {train_loss_topo:.4f}')
            print(f'  Val Dice: {val_dice:.4f} | Val IoU: {val_metrics["iou"]:.4f} | Val Prec: {val_metrics["precision"]:.4f} | Val Rec: {val_metrics["recall"]:.4f}')
            print(f'  CL-Break: {val_metrics.get("cl_break", 0):.1f} | Δβ₀: {val_metrics.get("delta_beta0", 0):.1f}')
            print(f'  Each Time: {self.format_time(epoch_time)} | Total Time: {self.format_time(elapsed)} | ETA: {self.format_time(eta)} | LR: {current_lr:.6f}')
            
            # 保存日志
            with open(self.log_file, 'a') as f:
                f.write(f'{self.current_epoch},{train_loss:.4f},{train_dice:.4f},'
                       f'{train_loss_topo:.4f},{val_dice:.4f},{val_metrics["iou"]:.4f},'
                       f'{val_metrics["precision"]:.4f},{val_metrics["recall"]:.4f},'
                       f'{val_metrics.get("cl_break", 0):.1f},{val_metrics.get("delta_beta0", 0):.1f},'
                       f'{current_lambda:.3f},{current_lr:.6f}\n')
            
            # 保存最佳模型
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                torch.save({
                    'epoch': self.current_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_dice': val_dice,
                }, self.checkpoint_dir / 'best_model_topo.pth')
            
            # 保存检查点
            if self.current_epoch % 10 == 0:
                torch.save({
                    'epoch': self.current_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}.pth')
        
        # 保存最终模型
        torch.save({
            'epoch': max_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.checkpoint_dir / 'final_model_topo.pth')
        
        # 训练结束
        total_time = (datetime.now() - self.start_time).total_seconds()
        print('\\n' + '=' * 80)
        print('训练完成')
        print(f'开始时间: {self.start_time.strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'结束时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'总用时: {total_time//60:.0f}m {total_time%60:.0f}s')
        print(f'平均每轮: {total_time/max_epochs:.0f}s')
        print(f'最佳验证Dice: {self.best_val_dice:.4f}')
        print(f'最佳模型: {self.checkpoint_dir / "best_model_topo.pth"}')
        print('=' * 80)


def main():
    """主函数。"""
    import argparse
    
    parser = argparse.ArgumentParser(description='端到端拓扑正则训练')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=None, help='批次大小')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点')
    args = parser.parse_args()
    
    # 加载配置
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 命令行参数覆盖配置
    if args.epochs:
        config['training']['max_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    # 创建trainer
    trainer = TrainerWithTopology(config, args)
    
    # 加载数据（从配置文件）
    # train_loader, val_loader, _ = get_drive_loaders()
    train_loader, val_loader, _ = get_combined_loaders(config)
    
    # 开始训练
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
