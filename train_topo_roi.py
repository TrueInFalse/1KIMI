# -*- coding: utf-8 -*-
"""
文件名: train_topo_roi.py
项目: retina_ph_seg
功能: 端到端拓扑正则训练（ROI对齐版）

与原版区别:
- 训练时Dice loss在ROI内计算（与验证口径一致）
- 其他Topo逻辑保持不变

作者: AI Assistant
创建日期: 2026-03-06
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
from model_unet import get_unet_model, count_parameters
from utils_metrics import (
    compute_basic_metrics,
    compute_topology_metrics,
    summarize_topology_results,
    tensor_to_numpy,
)
from topology_loss_fragment_suppress import TopologicalRegularizerFragmentSuppress


SUPPORTED_TOPOLOGY_VARIANTS = {
    'fragment_suppress',
    'thresholded_fs',
    'topk_fs',
}


def resolve_topology_variant(config: Dict) -> str:
    topo_cfg = config.get('topology', {})
    variant = topo_cfg.get('variant', 'fragment_suppress')
    if variant not in SUPPORTED_TOPOLOGY_VARIANTS:
        raise ValueError(
            f"Unsupported topology.variant={variant}. "
            f"Expected one of {sorted(SUPPORTED_TOPOLOGY_VARIANTS)}."
        )
    return variant


def compute_dice_loss_roi(pred_logits, target, roi_mask, eps=1e-7):
    """在ROI内计算Dice loss"""
    pred = torch.sigmoid(pred_logits)
    
    # 应用ROI mask
    pred = pred * roi_mask
    target = target * roi_mask
    
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    
    dice = (2.0 * intersection + eps) / (union + eps)
    loss = 1.0 - dice.mean()
    
    return loss


class LambdaScheduler:
    """λ课程学习调度器（与原版相同）"""
    
    def __init__(
        self,
        max_epochs: int = 200,
        strategy: str = '3175',
        warmup_epochs: int = 30,
        ramp_epochs: int = 70,
        warmup_lambda: float = 0.1,
        final_lambda: float = 0.5
    ):
        self.max_epochs = max_epochs
        self.strategy = strategy.lower()
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = max(1, ramp_epochs)
        self.warmup_lambda = warmup_lambda
        self.final_lambda = final_lambda
        self._print_schedule_info()

    def _print_schedule_info(self) -> None:
        if self.strategy == '0':
            print(f"[λ策略:0] λ恒为0（基线对照）")
        elif self.strategy == '3175':
            hold_epochs = max(0, self.max_epochs - self.warmup_epochs - self.ramp_epochs)
            print(f"[λ策略:3175] 总轮数{self.max_epochs}: 阶段1(固定{self.warmup_lambda})={self.warmup_epochs}轮, 阶段2(线性{self.warmup_lambda}→{self.final_lambda})={self.ramp_epochs}轮, 阶段3(固定{self.final_lambda})={hold_epochs}轮")
        else:
            raise ValueError(f"不支持的lambda策略: {self.strategy}")
    
    def get_lambda(self, epoch: int) -> float:
        if self.strategy == '0':
            return 0.0
        if epoch < self.warmup_epochs:
            return self.warmup_lambda
        if epoch < self.warmup_epochs + self.ramp_epochs:
            progress = (epoch - self.warmup_epochs) / self.ramp_epochs
            return self.warmup_lambda + progress * (self.final_lambda - self.warmup_lambda)
        return self.final_lambda


class EarlyStopping:
    """早停机制（与原版相同）"""
    
    def __init__(self, patience: int = 20, min_delta: float = 1e-4, mode: str = 'max'):
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


class TrainerWithTopologyROI:
    """带拓扑正则的训练器（ROI对齐版）"""
    
    def __init__(self, config: Dict, args: Optional[Any] = None):
        self.config = config
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        topo_cfg = config.get('topology', {})
        self.topology_variant = resolve_topology_variant(config)
        self.fragment_tau = float(topo_cfg.get('fragment_tau', 0.01))
        self.fragment_topk = max(0, int(topo_cfg.get('fragment_topk', 4)))

        train_cfg = config.get('training', {})
        self.checkpoint_dir = Path(train_cfg.get('checkpoint_dir', './checkpoints'))
        self.log_dir = Path(train_cfg.get('log_dir', './logs'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._setup_output_paths()
        
        self._setup_model()
        
        # 损失函数（主线 Fragment-Suppress 版）
        topo_cfg = config.get('topology', {})
        requested_loss_mode = getattr(args, 'loss_mode', 'fragment_suppress') if args else 'fragment_suppress'
        if requested_loss_mode != 'fragment_suppress':
            print(
                f"[兼容提示] --loss-mode={requested_loss_mode} 仅保留给旧入口；"
                f"当前实际使用 topology.variant={self.topology_variant}"
            )
        self.loss_mode = self.topology_variant
        self.criterion_topo = TopologicalRegularizerFragmentSuppress(
            target_beta0=topo_cfg.get('target_beta0', 5),
            max_death=topo_cfg.get('max_death', 0.5),
            loss_scale=topo_cfg.get('loss_scale', 100.0),
            fragment_penalty_factor=topo_cfg.get('fragment_penalty_factor', 1.0),
            loss_mode=self.loss_mode,
            target_lifetime=topo_cfg.get('target_lifetime', 0.5),
            main_boost_factor=topo_cfg.get('main_boost_factor', 1.0),
            variant=self.topology_variant,
            fragment_tau=self.fragment_tau,
            fragment_topk=self.fragment_topk,
        ).to(self.device)
        
        # λ调度器
        max_epochs = config['training']['max_epochs']
        lambda_cfg = topo_cfg.get('lambda_schedule', {})
        self.lambda_scheduler = LambdaScheduler(
            max_epochs=max_epochs,
            strategy=lambda_cfg.get('strategy', '3175'),
            warmup_epochs=lambda_cfg.get('warmup_epochs', 30),
            ramp_epochs=lambda_cfg.get('ramp_epochs', 70),
            warmup_lambda=lambda_cfg.get('warmup_lambda', 0.1),
            final_lambda=lambda_cfg.get('final_lambda', 0.5),
        )
        
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
        self._init_log(overwrite=True)
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_dice = 0.0
        self.start_time = None
        print(f"Topology Variant: {self.topology_variant}")
        if self.topology_variant == 'thresholded_fs':
            print(f"Fragment Tau: {self.fragment_tau}")
        elif self.topology_variant == 'topk_fs':
            print(f"Fragment TopK: {self.fragment_topk}")
        print(f"Topo Log File: {self.log_file}")
        print(f"Topo Best Checkpoint: {self.best_checkpoint_path}")
        
        print(f'设备: {self.device}')
        print('Topo Loss: see Topology Variant above (fragment_suppress remains the mainline default)')
    
    def _setup_model(self) -> None:
        """初始化模型"""
        model_cfg = self.config['model']
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
    
    def _setup_output_paths(self) -> None:
        variant_suffix = self.topology_variant
        self.log_file = self.log_dir / f'training_topo_roi_{variant_suffix}_log.csv'
        self.best_checkpoint_path = self.checkpoint_dir / f'best_model_topo_roi_{variant_suffix}.pth'
        self.final_checkpoint_path = self.checkpoint_dir / f'final_model_topo_roi_{variant_suffix}.pth'

        self.log_files = [self.log_file]
        self.best_checkpoint_paths = [self.best_checkpoint_path]
        self.final_checkpoint_paths = [self.final_checkpoint_path]

        if self.topology_variant == 'fragment_suppress':
            legacy_log = self.log_dir / 'training_topo_roi_log.csv'
            legacy_best = self.checkpoint_dir / 'best_model_topo_roi.pth'
            legacy_final = self.checkpoint_dir / 'final_model_topo_roi.pth'
            if legacy_log not in self.log_files:
                self.log_files.append(legacy_log)
            if legacy_best not in self.best_checkpoint_paths:
                self.best_checkpoint_paths.append(legacy_best)
            if legacy_final not in self.final_checkpoint_paths:
                self.final_checkpoint_paths.append(legacy_final)

    def _init_log(self, overwrite: bool = False) -> None:
        """初始化日志文件（ROI对齐版）"""
        header = (
            'epoch,train_loss,train_dice,train_dice_loss_roi,train_loss_topo,'
            'val_dice,val_iou,val_precision,val_recall,'
            'cl_break,delta_beta0,pred_beta0,target_beta0,max_lifetime,fragments_mean,'
            'topology_valid_count,topology_invalid_count,lambda,lr,'
            'topo_loss_raw,topo_loss_scaled,ratio,'
            'roi_mode,roi_mean\n'
        )
        for log_path in self.log_files:
            if overwrite or not log_path.exists():
                with open(log_path, 'w') as f:
                    f.write(header)
    
    def _append_log_line(self, line: str) -> None:
        for log_path in self.log_files:
            with open(log_path, 'a') as f:
                f.write(line)

    @staticmethod
    def _save_checkpoint_payload(paths: List[Path], payload: Dict[str, Any]) -> None:
        for checkpoint_path in paths:
            torch.save(payload, checkpoint_path)

    def format_time(self, seconds: float) -> str:
        """格式化时间"""
        if seconds < 60:
            return f'{seconds:.0f}s'
        elif seconds < 3600:
            return f'{seconds/60:.0f}m {seconds%60:.0f}s'
        else:
            return f'{seconds/3600:.0f}h {(seconds%3600)/60:.0f}m'
    
    @staticmethod
    def _normalize_roi_tensor(rois: Any, device: torch.device) -> torch.Tensor:
        """将ROI批次统一为 [B, 1, H, W]"""
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
    
    def train_epoch(self, train_loader: DataLoader, epoch_idx: int) -> Dict[str, float]:
        """训练一个epoch（ROI对齐版）"""
        self.model.train()
        
        total_loss = 0.0
        total_dice = 0.0
        total_dice_loss_roi = 0.0
        total_loss_topo = 0.0
        total_topo_raw = 0.0
        total_topo_scaled = 0.0
        total_ratio = 0.0
        total_roi_mean = 0.0
        pd_max_lifetimes: List[float] = []
        pd_fragments_means: List[float] = []
        last_pd_stats_sample: Optional[Dict[str, float]] = None
        
        num_batches = len(train_loader)
        current_lambda = self.lambda_scheduler.get_lambda(epoch_idx)
        loss_scale = getattr(self.criterion_topo, 'loss_scale', 1.0)
        
        for batch_idx, batch in enumerate(train_loader):
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                images, vessels, rois = batch
                images = images.to(self.device)
                vessels = vessels.to(self.device)
                rois = self._normalize_roi_tensor(rois, self.device)
            else:
                images = batch['image'].to(self.device)
                vessels = batch['vessel'].to(self.device)
                rois = self._normalize_roi_tensor(batch['roi'], self.device)
            
            # 统计ROI
            with torch.no_grad():
                roi_mean = rois[0, 0].mean().item()
                total_roi_mean += roi_mean
            
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(images)
            
            # Dice损失（ROI内计算 - 关键修改）
            loss_dice = compute_dice_loss_roi(outputs, vessels, rois)
            
            # 拓扑损失
            pred = torch.sigmoid(outputs)
            loss_topo = self.criterion_topo(pred, rois, self.current_epoch)
            batch_pd_stats = self.criterion_topo.get_last_pd_stats()
            if batch_pd_stats:
                last_pd_stats_sample = batch_pd_stats[0]
                for sample_stats in batch_pd_stats:
                    pd_max_lifetimes.append(float(sample_stats.get('max_lifetime', float('nan'))))
                    pd_fragments_means.append(float(sample_stats.get('fragments_mean', float('nan'))))
            
            # 计算各类损失值
            topo_raw = loss_topo.item() / loss_scale if loss_scale > 0 else loss_topo.item()
            topo_scaled = loss_topo.item() * current_lambda
            ratio = topo_scaled / (loss_dice.item() + 1e-8)
            
            # 总损失
            loss = loss_dice + current_lambda * loss_topo
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_loss_topo += loss_topo.item()
            total_dice_loss_roi += loss_dice.item()
            total_topo_raw += topo_raw
            total_topo_scaled += topo_scaled
            total_ratio += ratio
            
            with torch.no_grad():
                pred_binary = (pred > 0.5).float()
                # ROI内估算Dice
                intersection = (pred_binary * vessels * rois).sum()
                dice = (2 * intersection) / ((pred_binary * rois).sum() + (vessels * rois).sum() + 1e-7)
                total_dice += dice.item()
            
            self.global_step += 1
        
        finite_max_lifetimes = [value for value in pd_max_lifetimes if np.isfinite(value)]
        finite_fragments_means = [value for value in pd_fragments_means if np.isfinite(value)]
        train_max_lifetime = float(np.nanmean(finite_max_lifetimes)) if finite_max_lifetimes else float('nan')
        train_fragments_mean = float(np.nanmean(finite_fragments_means)) if finite_fragments_means else float('nan')
        
        stats = {
            'avg_loss': total_loss / num_batches,
            'avg_dice': total_dice / num_batches,
            'avg_dice_loss_roi': total_dice_loss_roi / num_batches,
            'avg_loss_topo': total_loss_topo / num_batches,
            'avg_topo_raw': total_topo_raw / num_batches,
            'avg_topo_scaled': total_topo_scaled / num_batches,
            'avg_ratio': total_ratio / num_batches,
            'roi_mean': total_roi_mean / num_batches,
            'train_max_lifetime': train_max_lifetime,
            'train_fragments_mean': train_fragments_mean,
            'pd_stats': last_pd_stats_sample,
        }
        
        return stats
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        
        metric_sums = {
            'dice': 0.0,
            'iou': 0.0,
            'precision': 0.0,
            'recall': 0.0,
        }
        all_topology_results: List[Dict[str, float]] = []
        num_samples = 0
        
        for batch in tqdm(val_loader, desc='Validate', leave=False):
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
            pred_batch = torch.sigmoid(outputs).cpu().numpy()
            mask_batch = vessels.cpu().numpy()
            roi_batch = rois.cpu().numpy()

            for pred_sample, mask_sample, roi_sample in zip(pred_batch, mask_batch, roi_batch):
                roi = roi_sample[0] if roi_sample.ndim == 3 else roi_sample

                basic_metrics = compute_basic_metrics(pred_sample[0], mask_sample[0], roi)
                for key in metric_sums:
                    metric_sums[key] += basic_metrics[key]

                try:
                    topo_m = compute_topology_metrics(pred_sample[0], mask_sample[0], roi)
                except Exception as exc:
                    topo_m = {
                        'cl_break': float('nan'),
                        'delta_beta0': float('nan'),
                        'pred_beta0': float('nan'),
                        'target_beta0': float('nan'),
                        'valid': False,
                        'error': str(exc),
                    }
                all_topology_results.append(topo_m)
                num_samples += 1

        topology_summary = summarize_topology_results(all_topology_results)

        if num_samples == 0:
            return {
                'dice': 0.0,
                'iou': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'cl_break': float('nan'),
                'delta_beta0': float('nan'),
                'pred_beta0': float('nan'),
                'target_beta0': float('nan'),
                'topology_valid_count': 0,
                'topology_invalid_count': 0,
            }
        
        metrics = {
            'dice': metric_sums['dice'] / num_samples,
            'iou': metric_sums['iou'] / num_samples,
            'precision': metric_sums['precision'] / num_samples,
            'recall': metric_sums['recall'] / num_samples,
            'cl_break': topology_summary.get('cl_break', float('nan')),
            'delta_beta0': topology_summary.get('delta_beta0', float('nan')),
            'pred_beta0': topology_summary.get('pred_beta0', float('nan')),
            'target_beta0': topology_summary.get('target_beta0', float('nan')),
            'topology_valid_count': topology_summary.get('valid_count', 0),
            'topology_invalid_count': topology_summary.get('invalid_count', 0),
        }
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """主训练循环"""
        self.start_time = datetime.now()
        max_epochs = self.config['training']['max_epochs']
        
        if self.enable_early_stopping:
            print(f'\n早停: 启用 (耐心值={self.config["training"]["patience"]})')
        else:
            print(f'\n早停: 禁用 (将跑满{max_epochs}轮)')
        
        print(f'开始训练 (Topo-ROI对齐版)')
        print(f'总轮数: {max_epochs}')
        print(f'关键修改: Dice loss在ROI内计算')
        print(f'开始时间: {self.start_time.strftime("%Y-%m-%d %H:%M:%S")}')
        print('=' * 80)
        
        roi_mode = self.config.get('data', {}).get('kaggle_roi', {}).get('mode', 'fov')
        
        for epoch in range(max_epochs):
            self.current_epoch = epoch + 1
            
            train_stats = self.train_epoch(train_loader, epoch)
            train_loss = train_stats['avg_loss']
            train_dice = train_stats['avg_dice']
            train_dice_loss_roi = train_stats['avg_dice_loss_roi']
            train_loss_topo = train_stats['avg_loss_topo']
            train_topo_raw = train_stats['avg_topo_raw']
            train_topo_scaled = train_stats['avg_topo_scaled']
            train_ratio = train_stats['avg_ratio']
            train_roi_mean = train_stats['roi_mean']
            train_max_lifetime = train_stats.get('train_max_lifetime', float('nan'))
            train_fragments_mean = train_stats.get('train_fragments_mean', float('nan'))
            
            val_metrics = self.validate(val_loader)
            val_dice = val_metrics['dice']
            
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            current_lambda = self.lambda_scheduler.get_lambda(epoch)
            
            elapsed = (datetime.now() - self.start_time).total_seconds()
            epoch_time = elapsed / self.current_epoch
            eta = epoch_time * (max_epochs - self.current_epoch)
            
            print(f'\nEpoch {self.current_epoch}/{max_epochs}  (λ={current_lambda:.3f})')
            print(f'  Train Dice Loss (ROI): {train_dice_loss_roi:.4f} | Train Dice (ROI): {train_dice:.4f} | Train Topo: {train_loss_topo:.4f}')
            print(f'  Val Dice: {val_dice:.4f} | Val IoU: {val_metrics["iou"]:.4f} | Val Prec: {val_metrics["precision"]:.4f} | Val Rec: {val_metrics["recall"]:.4f}')
            print(f'  CL-Break: {val_metrics.get("cl_break", 0):.1f} | Δβ₀: {val_metrics.get("delta_beta0", 0):.1f}')
            print(f'  β₀(pred/target): {val_metrics.get("pred_beta0", float("nan")):.1f}/{val_metrics.get("target_beta0", float("nan")):.1f} | TopoValid/Invalid: {int(val_metrics.get("topology_valid_count", 0))}/{int(val_metrics.get("topology_invalid_count", 0))}')
            print(f'  Train PD MaxLifetime: {train_max_lifetime:.4f} | Train PD FragmentsMean: {train_fragments_mean:.4f}')
            print(f'  Ratio: {train_ratio:.4f} | TopoScaled: {train_topo_scaled:.4f}')
            print(f'  ROI Mean: {train_roi_mean:.4f} | ROI Mode: {roi_mode}')
            print(f'  Topology Variant: {self.topology_variant}')
            if self.topology_variant == 'thresholded_fs':
                print(f'  Fragment Tau: {self.fragment_tau}')
            elif self.topology_variant == 'topk_fs':
                print(f'  Fragment TopK: {self.fragment_topk}')
            print(f'  Each Time: {self.format_time(epoch_time)} | Total Time: {self.format_time(elapsed)} | ETA: {self.format_time(eta)} | LR: {current_lr:.6f}')
            
            # 打印PD统计信息（主线诊断用）
            if self.current_epoch in [1, 5, 10, 20] and train_stats.get('pd_stats'):
                pd_stats = train_stats['pd_stats']
                print(f'\n  [PD Stats - Epoch {self.current_epoch}]')
                print(f'    Num Finite: {pd_stats["num_finite"]}')
                print(f'    Max Lifetime: {pd_stats["max_lifetime"]:.4f}')
                print(f'    Top5 Lifetimes: {[f"{x:.4f}" for x in pd_stats["top5_lifetimes"]]}')
                print(f'    Fragments Mean (excl. main): {pd_stats["fragments_mean"]:.4f}')
                if self.topology_variant == 'thresholded_fs':
                    print(
                        f'    Active Fragments Mean (>tau): {pd_stats.get("active_fragments_mean", 0.0):.4f} '
                        f'| Active Count: {int(pd_stats.get("active_fragments_count", 0))}'
                    )
                if self.topology_variant == 'topk_fs':
                    print(
                        f'    Selected TopK Mean: {pd_stats.get("selected_topk_mean", 0.0):.4f} '
                        f'| Selected Count: {int(pd_stats.get("selected_topk_count", 0))}'
                    )
            
            log_line = (
                f'{self.current_epoch},{train_loss:.4f},{train_dice:.4f},{train_dice_loss_roi:.6f},{train_loss_topo:.4f},'
                f'{val_dice:.4f},{val_metrics["iou"]:.4f},{val_metrics["precision"]:.4f},{val_metrics["recall"]:.4f},'
                f'{val_metrics.get("cl_break", float("nan")):.1f},{val_metrics.get("delta_beta0", float("nan")):.1f},'
                f'{val_metrics.get("pred_beta0", float("nan")):.1f},{val_metrics.get("target_beta0", float("nan")):.1f},'
                f'{train_max_lifetime:.6f},{train_fragments_mean:.6f},'
                f'{int(val_metrics.get("topology_valid_count", 0))},{int(val_metrics.get("topology_invalid_count", 0))},'
                f'{current_lambda:.3f},{current_lr:.6f},'
                f'{train_topo_raw:.6f},{train_topo_scaled:.6f},{train_ratio:.6f},'
                f'{roi_mode},{train_roi_mean:.4f}\n'
            )
            self._append_log_line(log_line)
            
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                self._save_checkpoint_payload(self.best_checkpoint_paths, {
                    'epoch': self.current_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_dice': val_dice,
                })
            
            if self.enable_early_stopping and self.early_stopping(val_dice, self.current_epoch):
                print(f'\n早停触发！最佳epoch: {self.early_stopping.best_epoch}, 最佳val_dice: {self.early_stopping.best_value:.4f}')
                break
        
        self._save_checkpoint_payload(self.final_checkpoint_paths, {
            'epoch': max_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        })
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        print('\n' + '=' * 80)
        print('训练完成')
        print(f'开始时间: {self.start_time.strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'结束时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'总用时: {total_time//60:.0f}m {total_time%60:.0f}s')
        print(f'最佳验证Dice: {self.best_val_dice:.4f}')
        print(f'最佳模型: {self.best_checkpoint_path}')
        print('=' * 80)


def set_seed(seed: int = 42, fast_dev: bool = False) -> Tuple[bool, bool]:
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    deterministic = not fast_dev
    benchmark = bool(fast_dev)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    return deterministic, benchmark


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='端到端拓扑正则训练（ROI对齐版）')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='配置文件路径（默认: config.yaml）')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数（仅显式传入时覆盖yaml中的training.max_epochs）')
    parser.add_argument('--fast-dev', action='store_true',
                        help='Disable deterministic cuDNN and enable benchmark for short diagnostic runs')
    parser.add_argument('--loss-mode', type=str, default='fragment_suppress',
                        choices=['standard', 'main_component', 'fragment_suppress'],
                        help='旧Topo入口兼容参数；实际变体请在config中设置 topology.variant')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    deterministic_mode, cudnn_benchmark = set_seed(
        config['training'].get('seed', 42),
        fast_dev=args.fast_dev,
    )

    if args.epochs is not None:
        config['training']['max_epochs'] = args.epochs

    requested_loss_mode = args.loss_mode
    topology_variant = resolve_topology_variant(config)
    fragment_tau = float(config.get('topology', {}).get('fragment_tau', 0.01))
    fragment_topk = max(0, int(config.get('topology', {}).get('fragment_topk', 4)))

    print(f"\n{'='*60}")
    print(f"Config: {args.config}")
    print(f"Max Epochs: {config['training']['max_epochs']} ({'CLI override' if args.epochs is not None else 'from yaml'})")
    print(f"Deterministic mode: {'ON' if deterministic_mode else 'OFF'}")
    print(f"cuDNN benchmark: {'ON' if cudnn_benchmark else 'OFF'}")
    print(f"Topology Variant: {topology_variant}")
    if topology_variant == 'fragment_suppress':
        print("Topology Variant Role: reference mainline")
    elif topology_variant == 'thresholded_fs':
        print(f"Fragment Tau: {fragment_tau}")
    elif topology_variant == 'topk_fs':
        print(f"Fragment TopK: {fragment_topk}")
    print(f"Legacy --loss-mode arg: {requested_loss_mode} (compatibility-only)")
    print(f"Checkpoint Dir: {config['training'].get('checkpoint_dir', './checkpoints')}")
    print(f"Log Dir: {config['training'].get('log_dir', './logs')}")
    print(f"{'='*60}\n")
    trainer = TrainerWithTopologyROI(config, args)
    train_loader, val_loader, _ = get_combined_loaders(config)
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
