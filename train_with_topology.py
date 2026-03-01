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
    """λ课程学习调度器（支持多策略可扩展）。

    当前内置策略:
    - ``015``:
      前30% epochs λ=0，接着30%线性0→0.1，最后40%线性0.1→0.5。
    - ``3175``:
      前30epoch固定0.1，后70epoch线性增至0.5，剩余epoch恒定0.5。
    """
    
    def __init__(
        self,
        max_epochs: int = 200,
        strategy: str = '015',
        phase1_ratio: float = 0.3,
        phase2_ratio: float = 0.3,
        phase2_end_lambda: float = 0.1,
        phase3_end_lambda: float = 0.5,
        warmup_epochs: int = 30,
        ramp_epochs: int = 70,
        warmup_lambda: float = 0.1,
        final_lambda: float = 0.5
    ):
        self.max_epochs = max_epochs
        self.strategy = strategy.lower()

        # 015策略参数
        self.phase1_epochs = int(max_epochs * phase1_ratio)
        self.phase2_epochs = int(max_epochs * phase2_ratio)
        self.phase3_epochs = max(1, max_epochs - self.phase1_epochs - self.phase2_epochs)
        self.phase2_end_lambda = phase2_end_lambda
        self.phase3_end_lambda = phase3_end_lambda

        # 3175策略参数
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = max(1, ramp_epochs)
        self.warmup_lambda = warmup_lambda
        self.final_lambda = final_lambda

        self._print_schedule_info()

    def _print_schedule_info(self) -> None:
        """打印当前调度策略摘要。"""
        if self.strategy == '015':
            print(
                f"[λ策略:015] 总轮数{self.max_epochs}: "
                f"阶段1(λ=0)={self.phase1_epochs}轮, "
                f"阶段2(0→{self.phase2_end_lambda})={self.phase2_epochs}轮, "
                f"阶段3({self.phase2_end_lambda}→{self.phase3_end_lambda})={self.phase3_epochs}轮"
            )
        elif self.strategy == '3175':
            hold_epochs = max(0, self.max_epochs - self.warmup_epochs - self.ramp_epochs)
            print(
                f"[λ策略:3175] 总轮数{self.max_epochs}: "
                f"阶段1(固定{self.warmup_lambda})={self.warmup_epochs}轮, "
                f"阶段2(线性{self.warmup_lambda}→{self.final_lambda})={self.ramp_epochs}轮, "
                f"阶段3(固定{self.final_lambda})={hold_epochs}轮"
            )
        else:
            raise ValueError(
                f"不支持的lambda策略: {self.strategy}。"
                f"可选: '015', '3175'"
            )
    
    def get_lambda(self, epoch: int) -> float:
        """获取当前epoch的λ值。"""
        if self.strategy == '015':
            if epoch < self.phase1_epochs:
                return 0.0
            if epoch < self.phase1_epochs + self.phase2_epochs:
                progress = (epoch - self.phase1_epochs) / max(1, self.phase2_epochs)
                return progress * self.phase2_end_lambda

            progress = (epoch - self.phase1_epochs - self.phase2_epochs) / self.phase3_epochs
            return self.phase2_end_lambda + progress * (self.phase3_end_lambda - self.phase2_end_lambda)

        # 3175策略
        if epoch < self.warmup_epochs:
            return self.warmup_lambda
        if epoch < self.warmup_epochs + self.ramp_epochs:
            progress = (epoch - self.warmup_epochs) / self.ramp_epochs
            return self.warmup_lambda + progress * (self.final_lambda - self.warmup_lambda)
        return self.final_lambda


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
        # 拓扑损失（经典稳定版）
        topo_cfg = config.get('topology', {})
        self.criterion_topo = CubicalRipserLoss(
            target_beta0=topo_cfg.get('target_beta0', 5),
            max_death=topo_cfg.get('max_death', 0.5),
            loss_scale=topo_cfg.get('loss_scale', 1.0)
        ).to(self.device)
        
        # λ调度器（多策略，可配置）
        max_epochs = config['training']['max_epochs']
        topo_cfg = config.get('topology', {})
        lambda_cfg = topo_cfg.get('lambda_schedule', {})
        self.lambda_scheduler = LambdaScheduler(
            max_epochs=max_epochs,
            strategy=lambda_cfg.get('strategy', '015'),
            phase1_ratio=lambda_cfg.get('phase1_ratio', 0.3),
            phase2_ratio=lambda_cfg.get('phase2_ratio', 0.3),
            phase2_end_lambda=lambda_cfg.get('phase2_end_lambda', 0.1),
            phase3_end_lambda=lambda_cfg.get('phase3_end_lambda', 0.5),
            warmup_epochs=lambda_cfg.get('warmup_epochs', 30),
            ramp_epochs=lambda_cfg.get('ramp_epochs', 70),
            warmup_lambda=lambda_cfg.get('warmup_lambda', 0.1),
            final_lambda=lambda_cfg.get('final_lambda', 0.5),
        )
        
        # 早停机制（通过enable_early_stopping开关控制，与基线模型统一）
        self.enable_early_stopping = config['training'].get('enable_early_stopping', True)
        if self.enable_early_stopping:
            from train_baseline import EarlyStopping
            self.early_stopping = EarlyStopping(
                patience=config['training']['patience'],
                mode='max'
            )
        else:
            self.early_stopping = None
        
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
    
    @staticmethod
    def _normalize_roi_tensor(rois: Any, device: torch.device) -> torch.Tensor:
        """将DataLoader返回的ROI批次统一为 [B, 1, H, W]。"""
        if isinstance(rois, torch.Tensor):
            roi_tensor = rois.to(device)
        elif isinstance(rois, (list, tuple)):
            roi_tensor = torch.stack([r.to(device) for r in rois])
        else:
            raise TypeError(f'不支持的ROI类型: {type(rois)}')

        if roi_tensor.dim() == 3:
            # [B, H, W] -> [B, 1, H, W]
            roi_tensor = roi_tensor.unsqueeze(1)
        elif roi_tensor.dim() == 4:
            # [B, 1, H, W] 已正确
            pass
        else:
            raise ValueError(f'ROI张量维度异常: {roi_tensor.shape}, 期望[B,H,W]或[B,1,H,W]')

        return roi_tensor.float()

    def train_epoch(self, train_loader: DataLoader, epoch_idx: int) -> Tuple[float, float, float]:
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
        
        # 获取当前λ（epoch_idx为0-based，避免与日志显示错位）
        current_lambda = self.lambda_scheduler.get_lambda(epoch_idx)
        
        # 是否开启 topo ROI debug（每个epoch只打印一次）
        debug_topo_roi = self.config.get('training', {}).get('debug_topo_roi', False)
        debug_printed = False
        
        for batch_idx, batch in enumerate(train_loader):
            # 处理list格式的batch [image, vessel, roi]
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                images, vessels, rois = batch
                images = images.to(self.device)
                vessels = vessels.to(self.device)
                rois = self._normalize_roi_tensor(rois, self.device)
            else:
                # dict格式（向后兼容）
                images = batch['image'].to(self.device)
                vessels = batch['vessel'].to(self.device)
                rois = self._normalize_roi_tensor(batch['roi'], self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(images)
            
            # Dice损失（始终计算）
            loss_dice = self.criterion_dice(outputs, vessels)
            
            # 拓扑损失（cripser 0.0.25+ 真正可微分！）
            pred = torch.sigmoid(outputs)
            
            loss_topo = self.criterion_topo(pred, rois, self.current_epoch)
            
            # Debug: 打印 ROI 和 prob_map 统计（每个 epoch 只打印一次）
            if debug_topo_roi and not debug_printed:
                with torch.no_grad():
                    roi_batch = rois[0, 0]  # 取第一个样本 [H, W]
                    prob_batch = pred[0, 0]  # 取第一个样本 [H, W]
                    
                    # ROI 统计
                    roi_mean = roi_batch.mean().item()
                    roi_sum = roi_batch.sum().item()
                    roi_unique = torch.unique(roi_batch).tolist()
                    
                    # prob 在 ROI 内外的统计
                    prob_in_roi = prob_batch[roi_batch == 1].mean().item() if (roi_batch == 1).any() else 0.0
                    prob_out_roi = prob_batch[roi_batch == 0].mean().item() if (roi_batch == 0).any() else 0.0
                    
                    # filtration = 1 - prob
                    fil_in_roi = (1 - prob_batch)[roi_batch == 1].mean().item() if (roi_batch == 1).any() else 0.0
                    fil_out_roi = (1 - prob_batch)[roi_batch == 0].mean().item() if (roi_batch == 0).any() else 0.0
                    
                    # 获取 loss_scale
                    loss_scale = getattr(self.criterion_topo, 'loss_scale', 1.0)
                    
                    print(f"\n[TopoROI-Debug Epoch {epoch_idx+1}]")
                    print(f"  ROI: mean={roi_mean:.4f}, sum={roi_sum:.0f}, unique={roi_unique}")
                    print(f"  Prob: in_roi={prob_in_roi:.4f}, out_roi={prob_out_roi:.4f}, diff={prob_in_roi-prob_out_roi:.4f}")
                    print(f"  Filtration: in_roi={fil_in_roi:.4f}, out_roi={fil_out_roi:.4f}")
                    print(f"  Loss: raw={loss_topo.item():.6f}, scaled={loss_topo.item():.6f}, lambda={current_lambda:.4f}, lambda*scale={current_lambda*loss_scale:.4f}")
                debug_printed = True
            
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
                rois = self._normalize_roi_tensor(rois, self.device)
            else:
                images = batch['image'].to(self.device)
                vessels = batch['vessel'].to(self.device)
                rois = self._normalize_roi_tensor(batch['roi'], self.device)
            
            outputs = self.model(images)
            pred = torch.sigmoid(outputs)
            
            all_preds.append(pred)
            all_masks.append(vessels)
            all_rois.append(rois)
        
        # 拼接所有批次并转为numpy
        all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
        all_masks = torch.cat(all_masks, dim=0).cpu().numpy()
        all_rois = torch.cat(all_rois, dim=0).cpu().numpy()
        
        # 逐个样本计算指标并取平均
        all_dice, all_iou, all_prec, all_rec = [], [], [], []
        all_cl_break, all_delta_beta0 = [], []
        
        for i in range(len(all_preds)):
            roi = all_rois[i, 0] if all_rois.ndim == 4 else all_rois[i]
            
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
            except Exception:
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
        
        
        # 打印早停状态
        if self.enable_early_stopping:
            print(f'\n早停: 启用 (耐心值={self.config["training"]["patience"]})')
        else:
            print(f'\n早停: 禁用 (将跑满{max_epochs}轮)')
        
        print(f'开始训练 (cripser 0.0.25+ 经典稳定版)')
        print(f'总轮数: {max_epochs}')
        print(f'开始时间: {self.start_time.strftime("%Y-%m-%d %H:%M:%S")}')
        print('=' * 80)
        
        for epoch in range(max_epochs):
            self.current_epoch = epoch + 1
            
            # 训练
            train_loss, train_dice, train_loss_topo = self.train_epoch(train_loader, epoch)
            
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
            print(f'  Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | Train Topo: {train_loss_topo:.4f} (含loss_scale)')
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
            
            # 早停检查（与基线模型统一开关）
            if self.enable_early_stopping and self.early_stopping(val_dice, self.current_epoch):
                print(f'\n早停触发！最佳epoch: {self.early_stopping.best_epoch}, '
                      f'最佳val_dice: {self.early_stopping.best_value:.4f}')
                break
        
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
