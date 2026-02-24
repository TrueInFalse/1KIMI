"""
拓扑正则化模块 - TopologicalRegularizer
基于cripser 0.0.25+真正可微分持续同调

关键特性：
- 子水平集filtration: 1.0 - prob_map（前景早birth）
- 纯2D张量输入: [H,W]，保持计算图完整
- 使用finite_lifetimes计算损失，无软近似
- 支持ROI掩码，背景强制1.0
"""

import torch
import torch.nn as nn
import cripser


class TopologicalRegularizer(nn.Module):
    """
    真正可微分拓扑正则化器（cripser 0.0.25+原生版）
    
    Args:
        target_beta0: 目标连通分量数（默认12）
        max_death: filtration域中最大death值（对应prob=0.5）
    """
    
    def __init__(self, target_beta0: int = 12, max_death: float = 0.5):
        super().__init__()
        self.target_beta0 = target_beta0
        self.max_death = max_death
    
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
        
        # 如果batch=1且dim=3，已经是[H,W]，无需squeeze
        # 如果batch>1或dim=4，需要遍历batch
        if prob_map.dim() == 4:
            # [B, 1, H, W] -> [B, H, W]
            prob_map = prob_map.squeeze(1)
        
        if roi_mask is not None and roi_mask.dim() == 4:
            roi_mask = roi_mask.squeeze(1)
        
        losses = []
        
        for i in range(batch_size):
            prob = prob_map[i]  # [H, W]
            
            # 应用ROI：背景区域prob强制为0（filtration=1.0）
            if roi_mask is not None:
                roi = roi_mask[i]
                prob = prob * roi + (1.0 - roi) * 0.0  # 背景prob=0
            
            # 关键：子水平集filtration，高prob -> 低filtration（早birth）
            filtration = 1.0 - prob
            
            # 确保2D且保持梯度
            if filtration.dim() > 2:
                filtration = filtration.squeeze()
            
            # 关键：直接传入torch.Tensor，保持requires_grad
            # compute_ph_torch原生支持torch autograd
            pd = cripser.compute_ph_torch(
                filtration,
                maxdim=0,
                filtration="V"
            )
            
            # 提取0维birth/death（filtration域）
            # pd: [N, 9] -> [dim, birth, death, ...]
            dim0_mask = pd[:, 0] == 0
            births = pd[dim0_mask, 1]
            deaths = pd[dim0_mask, 2]
            
            # 使用finite_lifetimes：过滤inf，计算持续
            finite_mask = torch.isfinite(deaths)
            if finite_mask.sum() == 0:
                losses.append(torch.tensor(0.0, device=device))
                continue
            
            births_finite = births[finite_mask]
            deaths_finite = deaths[finite_mask]
            lifetimes = deaths_finite - births_finite
            
            # 关键：只保留前target_beta0个最长持续的组件
            if len(lifetimes) > self.target_beta0:
                lifetimes, _ = torch.topk(lifetimes, self.target_beta0)
            
            # 损失：鼓励所有组件都有长持续（接近max_death=0.5）
            # MSE到目标lifetime=0.5
            target_lifetime = 0.5
            loss_i = torch.nn.functional.mse_loss(lifetimes, torch.full_like(lifetimes, target_lifetime))
            
            losses.append(loss_i)
        
        return torch.stack(losses).mean()


# 向后兼容别名
CubicalRipserLoss = TopologicalRegularizer
