"""
拓扑正则化模块 - TopologicalRegularizer（加强版v3.0）
基于cripser 0.0.25+真正可微分持续同调

关键特性：
- 自适应target_beta0：从训练集金标计算平均Betti数
- 新损失公式：excess惩罚(30%) + short持续惩罚(70%)
- 动态loss_scale：可在config.yaml配置（默认300.0）
"""

import torch
import torch.nn as nn
import cripser
import numpy as np
from typing import Optional, Dict


class TopologicalRegularizer(nn.Module):
    """
    真正可微分拓扑正则化器（cripser 0.0.25+原生版）
    
    Args:
        target_beta0: 目标连通分量数（默认None，自动从训练集计算）
        max_death: filtration域中最大death值（对应prob=0.5）
        loss_scale: 损失缩放因子（默认300.0）
        excess_weight: excess惩罚权重（默认0.3）
        short_weight: short持续惩罚权重（默认0.7）
        short_threshold: 短持续阈值（默认0.08）
    """
    
    def __init__(
        self, 
        target_beta0: Optional[int] = None, 
        max_death: float = 0.5,
        loss_scale: float = 300.0,
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
        
        # 如果未提供target_beta0，标记为需要计算
        self._auto_compute_beta0 = target_beta0 is None
        
        if target_beta0 is not None:
            print(f"拓扑损失: target_beta0={target_beta0} (手动设置)")
        else:
            print("拓扑损失: target_beta0将在首次forward时自动计算")
    
    def compute_target_beta0_from_loader(self, train_loader) -> int:
        """
        从训练集金标计算平均0维Betti数
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            target_beta0: 平均连通分量数（取整）
        """
        print("计算target_beta0: 遍历训练集金标...")
        betti_numbers = []
        
        # 遍历训练集（最多100个样本，避免太慢）
        max_samples = min(100, len(train_loader.dataset))
        sample_count = 0
        
        for batch in train_loader:
            if sample_count >= max_samples:
                break
                
            # 获取金标mask [B, 1, H, W] 或 [B, H, W]
            if isinstance(batch, (list, tuple)):
                vessels = batch[1]  # 假设第二个元素是vessel mask
            else:
                vessels = batch['vessel']
            
            # 转为numpy并计算每张图的连通分量数
            vessels_np = vessels.cpu().numpy()
            
            for i in range(vessels_np.shape[0]):
                if sample_count >= max_samples:
                    break
                    
                mask = vessels_np[i]
                if mask.ndim == 3:
                    mask = mask[0]  # [1, H, W] -> [H, W]
                
                # 二值化
                binary_mask = (mask > 0.5).astype(np.uint8)
                
                # 使用skimage计算连通分量数（0维Betti数）
                try:
                    from skimage.measure import label
                    labeled = label(binary_mask, connectivity=1)
                    num_components = labeled.max()
                    betti_numbers.append(num_components)
                    sample_count += 1
                except ImportError:
                    # 如果没有skimage，使用简单的计数
                    print("警告: 未安装skimage，使用默认target_beta0=4")
                    return 4
        
        if len(betti_numbers) == 0:
            print("警告: 未计算到Betti数，使用默认值4")
            return 4
        
        avg_beta0 = int(np.mean(betti_numbers))
        print(f"  样本数: {len(betti_numbers)}, 平均Betti数: {avg_beta0}")
        print(f"  Betti数范围: [{min(betti_numbers)}, {max(betti_numbers)}]")
        
        return max(1, avg_beta0)  # 至少为1
    
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
        num_bars_list = []  # 用于统计
        
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
            pd = cripser.compute_ph_torch(
                filtration,
                maxdim=0,
                filtration="V"
            )
            
            # 提取0维birth/death
            dim0_mask = pd[:, 0] == 0
            births = pd[dim0_mask, 1]
            deaths = pd[dim0_mask, 2]
            
            # 使用finite_lifetimes
            finite_mask = torch.isfinite(deaths)
            if finite_mask.sum() == 0:
                losses.append(torch.tensor(0.0, device=device))
                continue
            
            births_finite = births[finite_mask]
            deaths_finite = deaths[finite_mask]
            lifetimes = deaths_finite - births_finite
            
            num_bars = len(lifetimes)
            num_bars_list.append(num_bars)
            
            # 新损失公式（加强版）
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
            'short_threshold': self.short_threshold,
            'auto_compute': self._auto_compute_beta0
        }


# 向后兼容别名
CubicalRipserLoss = TopologicalRegularizer
