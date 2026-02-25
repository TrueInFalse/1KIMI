"""
拓扑正则化模块 - TopologicalRegularizer（稳定版v3.1）
基于cripser 0.0.25+真正可微分持续同调

关键特性：
- 固定target_beta0=5（经验值，适合血管树结构）
- 稳定loss_scale=10.0（避免数值爆炸）
- 新损失公式：excess惩罚(30%) + short持续惩罚(70%)
"""

import torch
import torch.nn as nn
import cripser
from typing import Dict


class TopologicalRegularizer(nn.Module):
    """
    真正可微分拓扑正则化器（cripser 0.0.25+原生版）
    
    Args:
        target_beta0: 目标连通分量数（默认5，经验值适合血管树）
        max_death: filtration域中最大death值（默认0.5）
        loss_scale: 损失缩放因子（默认10.0，稳定不爆炸）
        excess_weight: excess惩罚权重（默认0.3）
        short_weight: short持续惩罚权重（默认0.7）
        short_threshold: 短持续阈值（默认0.08）
    """
    
    def __init__(
        self, 
        target_beta0: int = 5, 
        max_death: float = 0.5,
        loss_scale: float = 10.0,
        excess_weight: float = 0.3,
        short_weight: float = 0.7,
        short_threshold: float = 0.08
    ):
        super().__init__()
        self.target_beta0 = target_beta0
        self.max_death = max_death
        self.loss_scale = loss_scale
        self.excess_weight = excess_weight
        self.short_weight = short_weight
        self.short_threshold = short_threshold
        
        print(f"[拓扑损失] target_beta0={target_beta0}, loss_scale={loss_scale}, "
              f"excess={excess_weight}, short={short_weight}")
    
    def forward(
        self, 
        prob_map: torch.Tensor, 
        roi_mask: torch.Tensor = None,
        epoch: int = None
    ) -> torch.Tensor:
        """
        计算拓扑正则损失
        
        Args:
            prob_map: [B, 1, H, W] 概率图（已sigmoid）
            roi_mask: [B, 1, H, W] ROI掩码（float32/64）
            epoch: 当前epoch（用于外部验证，本类内部不使用）
            
        Returns:
            loss: 标量Tensor（带grad_fn）
        """
        batch_size = prob_map.shape[0]
        device = prob_map.device
        
        if prob_map.dim() == 4:
            prob_map = prob_map.squeeze(1)
        
        if roi_mask is not None and roi_mask.dim() == 4:
            roi_mask = roi_mask.squeeze(1)
        
        losses = []
        
        for i in range(batch_size):
            prob = prob_map[i]
            
            # 应用ROI
            if roi_mask is not None:
                roi = roi_mask[i]
                prob = prob * roi + (1.0 - roi) * 0.0
            
            # 子水平集filtration
            filtration = 1.0 - prob
            
            if filtration.dim() > 2:
                filtration = filtration.squeeze()
            
            # 计算持续同调
            pd = cripser.compute_ph_torch(
                filtration,
                maxdim=0,
                filtration="V"
            )
            
            # 提取0维birth/death
            dim0_mask = pd[:, 0] == 0
            births = pd[dim0_mask, 1]
            deaths = pd[dim0_mask, 2]
            
            # 过滤inf
            finite_mask = torch.isfinite(deaths)
            if finite_mask.sum() == 0:
                losses.append(torch.tensor(0.0, device=device))
                continue
            
            births_finite = births[finite_mask]
            deaths_finite = deaths[finite_mask]
            lifetimes = deaths_finite - births_finite
            
            num_bars = len(lifetimes)
            
            # 新损失公式（稳定版）
            # 1. 惩罚过多的连通分量（超过target_beta0）
            excess = torch.relu(torch.tensor(float(num_bars), device=device) - self.target_beta0)
            excess_penalty = (excess ** 2) * self.excess_weight
            
            # 2. 惩罚短持续的组件（lifetimes < short_threshold）
            short_penalty = torch.relu(self.short_threshold - lifetimes).sum() * self.short_weight
            
            # 总损失
            loss_i = excess_penalty + short_penalty
            losses.append(loss_i)
        
        # 计算平均损失并缩放
        mean_loss = torch.stack(losses).mean()
        scaled_loss = mean_loss * self.loss_scale
        
        return scaled_loss
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'target_beta0': self.target_beta0,
            'loss_scale': self.loss_scale,
            'excess_weight': self.excess_weight,
            'short_weight': self.short_weight,
            'short_threshold': self.short_threshold
        }


# 向后兼容别名
CubicalRipserLoss = TopologicalRegularizer
