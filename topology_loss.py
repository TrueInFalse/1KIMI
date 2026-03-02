"""
拓扑正则化模块 - TopologicalRegularizer（经典稳定版 + Hinge版）
基于cripser 0.0.25+真正可微分持续同调

经典稳定公式：
- 保留前target_beta0个最长持续组件
- MSE损失到目标lifetime=0.5
- 简单稳定，无数值爆炸风险

Hinge版（任务3防御性loss）：
- 只惩罚 lifetimes < target_lifetime，不惩罚更长
- 防止模型为迎合MSE产生极端行为
"""

import torch
import torch.nn as nn
import cripser


class TopologicalRegularizer(nn.Module):
    """
    经典拓扑正则化器（稳定版 + Hinge版）
    
    Args:
        target_beta0: 目标连通分量数（默认5，经验值适合血管树）
        max_death: filtration域中最大death值（默认0.5）
        loss_scale: 损失缩放因子（默认1.0，通过实验调整）
        loss_mode: 'mse' 或 'hinge'（默认'mse'）
    """
    
    def __init__(self, target_beta0: int = 5, max_death: float = 0.5, loss_scale: float = 1.0, loss_mode: str = 'mse'):
        super().__init__()
        self.target_beta0 = target_beta0
        self.max_death = max_death
        self.loss_scale = loss_scale
        self.loss_mode = loss_mode
        print(f"[拓扑损失-{loss_mode}版] target_beta0={target_beta0}, max_death={max_death}, loss_scale={loss_scale}")
    
    def forward(
        self, 
        prob_map: torch.Tensor, 
        roi_mask: torch.Tensor = None,
        epoch: int = None
    ) -> torch.Tensor:
        """
        计算拓扑正则损失（经典MSE公式）
        
        Args:
            prob_map: [B, 1, H, W] 概率图（已sigmoid）
            roi_mask: [B, 1, H, W] ROI掩码
            
        Returns:
            loss: 标量Tensor
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
            
            # 经典公式：只保留前target_beta0个最长持续的组件
            if len(lifetimes) > self.target_beta0:
                lifetimes, _ = torch.topk(lifetimes, self.target_beta0)
            
            # 计算损失
            target_lifetime = 0.5
            if self.loss_mode == 'hinge':
                # Hinge版：只惩罚 lifetimes < target，不惩罚更长
                # 鼓励连通性，但不强制限制最大lifetime
                loss_i = torch.nn.functional.relu(target_lifetime - lifetimes).pow(2).mean()
            else:
                # MSE版：经典公式，惩罚到目标lifetime的偏差
                loss_i = torch.nn.functional.mse_loss(
                    lifetimes, 
                    torch.full_like(lifetimes, target_lifetime)
                )
            
            losses.append(loss_i)
        
        # 应用损失缩放
        return torch.stack(losses).mean() * self.loss_scale


# 向后兼容别名
CubicalRipserLoss = TopologicalRegularizer
