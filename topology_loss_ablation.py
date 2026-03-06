"""
拓扑损失目标形状消融实验模块

三种 loss 模式：
1. 'standard': 当前实现 - top-k + MSE(target_lifetime=0.5)
2. 'main_component': 主连通分量增强版 - 只奖励最长的 0维 finite lifetime 变大
3. 'fragment_suppress': 碎片抑制版 - 只惩罚除最长条外的 finite lifetimes 过大

作者: AI Assistant
创建日期: 2026-03-06
"""

import torch
import torch.nn as nn
import cripser


class TopologicalRegularizerAblation(nn.Module):
    """
    拓扑正则化器（消融实验版）
    
    Args:
        target_beta0: 目标连通分量数（仅用于standard模式）
        max_death: filtration域中最大death值
        loss_scale: 损失缩放因子
        loss_mode: 'standard', 'main_component', 或 'fragment_suppress'
        target_lifetime: 目标lifetime（standard模式用）
        main_boost_factor: 主分量奖励系数（main_component模式用）
        fragment_penalty_factor: 碎片惩罚系数（fragment_suppress模式用）
    """
    
    def __init__(
        self, 
        target_beta0: int = 5, 
        max_death: float = 0.5, 
        loss_scale: float = 100.0,
        loss_mode: str = 'standard',
        target_lifetime: float = 0.5,
        main_boost_factor: float = 1.0,
        fragment_penalty_factor: float = 1.0
    ):
        super().__init__()
        self.target_beta0 = target_beta0
        self.max_death = max_death
        self.loss_scale = loss_scale
        self.loss_mode = loss_mode
        self.target_lifetime = target_lifetime
        self.main_boost_factor = main_boost_factor
        self.fragment_penalty_factor = fragment_penalty_factor
        
        print(f"[拓扑损失-消融] mode={loss_mode}, target_beta0={target_beta0}, loss_scale={loss_scale}")
        if loss_mode == 'main_component':
            print(f"  main_boost_factor={main_boost_factor}")
        elif loss_mode == 'fragment_suppress':
            print(f"  fragment_penalty_factor={fragment_penalty_factor}")
    
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
        pd_stats = []  # 用于统计PD信息
        
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
                pd_stats.append({
                    'num_finite': 0,
                    'max_lifetime': 0.0,
                    'top5_lifetimes': [],
                    'fragments_mean': 0.0
                })
                continue
            
            births_finite = births[finite_mask]
            deaths_finite = deaths[finite_mask]
            lifetimes = deaths_finite - births_finite
            
            # 按lifetime降序排序
            sorted_lifetimes, _ = torch.sort(lifetimes, descending=True)
            
            # 收集PD统计信息
            num_finite = len(sorted_lifetimes)
            max_lifetime = sorted_lifetimes[0].item() if num_finite > 0 else 0.0
            top5 = [sorted_lifetimes[i].item() if i < num_finite else 0.0 for i in range(min(5, num_finite))]
            fragments_mean = sorted_lifetimes[1:].mean().item() if num_finite > 1 else 0.0
            
            pd_stats.append({
                'num_finite': num_finite,
                'max_lifetime': max_lifetime,
                'top5_lifetimes': top5,
                'fragments_mean': fragments_mean
            })
            
            # 根据模式计算loss
            if self.loss_mode == 'standard':
                loss_i = self._standard_loss(sorted_lifetimes)
            elif self.loss_mode == 'main_component':
                loss_i = self._main_component_loss(sorted_lifetimes)
            elif self.loss_mode == 'fragment_suppress':
                loss_i = self._fragment_suppress_loss(sorted_lifetimes)
            else:
                raise ValueError(f"Unknown loss_mode: {self.loss_mode}")
            
            losses.append(loss_i)
        
        # 保存PD统计到模块属性（供外部读取）
        self.last_pd_stats = pd_stats
        
        # 应用损失缩放
        return torch.stack(losses).mean() * self.loss_scale
    
    def _standard_loss(self, sorted_lifetimes):
        """
        当前标准实现: top-k + MSE(target_lifetime=0.5)
        鼓励前target_beta0个组件的lifetime都接近0.5
        """
        # 只保留前target_beta0个最长持续的组件
        if len(sorted_lifetimes) > self.target_beta0:
            lifetimes = sorted_lifetimes[:self.target_beta0]
        else:
            lifetimes = sorted_lifetimes
        
        # MSE到目标lifetime
        target = torch.full_like(lifetimes, self.target_lifetime)
        loss = torch.nn.functional.mse_loss(lifetimes, target)
        
        return loss
    
    def _main_component_loss(self, sorted_lifetimes):
        """
        主连通分量增强版: 只奖励最长的0维finite lifetime变大
        不再把top-k全部拉向同一个target
        
        策略:
        - 只关注最长的lifetime（主连通分量）
        - 鼓励它尽可能大（接近1.0）
        - 其他lifetimes不施加奖励/惩罚
        """
        if len(sorted_lifetimes) == 0:
            return torch.tensor(0.0, device=sorted_lifetimes.device)
        
        # 只取最长的lifetime
        main_lifetime = sorted_lifetimes[0]
        
        # 鼓励主lifetime变大：用 (1 - lifetime)^2 作为loss（越小越好）
        # 当lifetime接近1时，loss接近0
        loss = (1.0 - main_lifetime).pow(2) * self.main_boost_factor
        
        return loss
    
    def _fragment_suppress_loss(self, sorted_lifetimes):
        """
        碎片抑制版: 不奖励top-k变大，只惩罚除最长条之外的finite lifetimes过大
        
        策略:
        - 目标是压制碎片连通分量长期存在
        - 对除最长条外的所有lifetimes施加惩罚（鼓励它们变小）
        - 不奖励主lifetime变大（由Dice loss负责）
        """
        if len(sorted_lifetimes) <= 1:
            return torch.tensor(0.0, device=sorted_lifetimes.device)
        
        # 取除最长条外的所有lifetimes（碎片）
        fragment_lifetimes = sorted_lifetimes[1:]
        
        # 惩罚碎片lifetime过大：用 lifetime^2 作为loss
        # 当fragment lifetime接近0时，loss接近0
        # 只惩罚大的fragment lifetime，不惩罚小的
        loss = fragment_lifetimes.pow(2).mean() * self.fragment_penalty_factor
        
        return loss
    
    def get_last_pd_stats(self):
        """获取最近一次forward的PD统计信息"""
        return getattr(self, 'last_pd_stats', [])


# 向后兼容：标准版
CubicalRipserLoss = TopologicalRegularizerAblation
