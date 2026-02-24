# -*- coding: utf-8 -*-
"""
文件名: utils_metrics.py
项目: retina_ph_seg
功能: 分割评估指标（含拓扑指标CL-Break和Δβ₀）
作者: AI Assistant
创建日期: 2026-02-07
Python版本: 3.12+

依赖关系:
  - 上游: 无
  - 下游: train_baseline.py（训练时日志）, evaluate.py（评估）

主要类/函数:
  - compute_basic_metrics(): 计算Dice/IoU/Precision/Recall（ROI内）
  - compute_topology_metrics(): 计算CL-Break和Δβ₀（ROI内）
  - MetricsTracker(): 指标追踪器

更新记录:
  - v1.0: 初始版本，实现基础指标和拓扑指标
  - v1.1: 修正Betti数计算，添加ROI掩码约束
"""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from skimage.morphology import skeletonize, opening, disk
from scipy import ndimage


@dataclass
class MetricsResult:
    """指标计算结果容器。"""
    # 基础分割指标（均在ROI内计算）
    dice: float = 0.0
    iou: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    
    # 拓扑指标（仅在ROI区域内计算）
    cl_break: float = 0.0      # 中心线碎片数（越高越差）
    delta_beta0: float = 0.0   # Betti₀误差（连通分量数差异）
    
    # 辅助信息
    valid: bool = True         # 指标是否有效
    message: str = ""          # 异常信息


def apply_roi_mask(
    pred: np.ndarray,
    target: np.ndarray,
    roi_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """应用ROI掩码，只保留ROI内的像素参与计算。
    
    Args:
        pred: 预测结果 [H, W]，二值或概率
        target: 真实标签 [H, W]，二值
        roi_mask: ROI掩码 [H, W]，二值
        
    Returns:
        pred_roi: ROI内的预测
        target_roi: ROI内的真实标签
    """
    # 确保形状一致
    assert pred.shape == target.shape == roi_mask.shape, \
        f"形状不匹配: pred{pred.shape}, target{target.shape}, roi{roi_mask.shape}"
    
    # 应用ROI掩码
    pred_roi = pred[roi_mask > 0]
    target_roi = target[roi_mask > 0]
    
    return pred_roi, target_roi


def compute_basic_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    threshold: float = 0.5
) -> Dict[str, float]:
    """计算基础分割指标（Dice、IoU、Precision、Recall）。
    
    所有指标均在ROI区域内计算（如果提供了roi_mask）。
    
    Args:
        pred: 预测结果 [H, W]，值域[0,1]（概率）或{0,1}（二值）
        target: 真实标签 [H, W]，值域{0,1}
        roi_mask: ROI掩码 [H, W]，值域{0,1}（可选）
        threshold: 二值化阈值
        
    Returns:
        metrics: 包含dice, iou, precision, recall的字典
        
    Example:
        >>> pred = np.random.rand(512, 512)
        >>> target = np.random.randint(0, 2, (512, 512))
        >>> roi = np.random.randint(0, 2, (512, 512))
        >>> metrics = compute_basic_metrics(pred, target, roi)
    """
    # 二值化预测
    pred_binary = (pred > threshold).astype(np.float32)
    target_binary = target.astype(np.float32)
    
    # 应用ROI掩码
    if roi_mask is not None:
        pred_binary, target_binary = apply_roi_mask(pred_binary, target_binary, roi_mask)
    
    # 处理边界情况
    if len(pred_binary) == 0:
        return {'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    # 计算混淆矩阵元素
    tp = np.sum((pred_binary == 1) & (target_binary == 1))
    fp = np.sum((pred_binary == 1) & (target_binary == 0))
    fn = np.sum((pred_binary == 0) & (target_binary == 1))
    tn = np.sum((pred_binary == 0) & (target_binary == 0))
    
    # 计算指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Dice = 2*TP / (2*TP + FP + FN)
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    
    # IoU = TP / (TP + FP + FN)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    
    return {
        'dice': float(dice),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall)
    }


def compute_betti0(image: np.ndarray) -> int:
    """计算Betti数β₀（连通分量数）。
    
    使用scipy.ndimage.label进行连通区域标记。
    
    Args:
        image: 二值图像 [H, W]
        
    Returns:
        beta0: 连通分量数（不包括背景）
    """
    # 标记连通区域
    labeled_array, num_features = ndimage.label(image)
    return num_features


def compute_betti0_filtered(image: np.ndarray, min_size: int = 20) -> int:
    """计算过滤后的Betti数β₀（忽略小组件）。
    
    过滤掉小于min_size像素的连通分量，避免噪声干扰。
    这是与拓扑损失训练目标一致的量纲。
    
    Args:
        image: 二值图像 [H, W]
        min_size: 最小像素数，小于此值的组件被忽略
        
    Returns:
        beta0: 过滤后的连通分量数
    """
    labeled_array, num_features = ndimage.label(image)
    
    if num_features == 0:
        return 0
    
    # 统计每个组件的大小
    valid_components = 0
    for i in range(1, num_features + 1):
        component_size = np.sum(labeled_array == i)
        if component_size >= min_size:
            valid_components += 1
    
    return valid_components


def skeletonize_vessel(
    vessel_mask: np.ndarray,
    roi_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """对血管掩码进行骨架化。
    
    使用Lee算法的骨架化，并在骨架化前进行形态学开运算去除小噪声。
    
    Args:
        vessel_mask: 血管二值掩码 [H, W]
        roi_mask: ROI掩码（可选）
        
    Returns:
        skeleton: 骨架化结果 [H, W]
    """
    # 确保二值
    vessel_binary = (vessel_mask > 0).astype(np.uint8)
    
    # 应用ROI约束（ROI外设为0）
    if roi_mask is not None:
        vessel_binary = vessel_binary * (roi_mask > 0).astype(np.uint8)
    
    if vessel_binary.sum() == 0:
        return np.zeros_like(vessel_binary)
    
    # 形态学开运算：去除小噪声（半径2的圆盘）
    # 这有助于减少CL-Break中的虚假碎片
    vessel_binary = opening(vessel_binary, disk(2))
    
    # 骨架化（使用Lee算法，保持拓扑结构）
    skeleton = skeletonize(vessel_binary, method='lee')
    
    return skeleton.astype(np.uint8)


def count_skeleton_fragments(
    skeleton: np.ndarray,
    min_length: int = 10
) -> int:
    """计算骨架碎片数（CL-Break）。
    
    骨架化后的连通分量数，过滤掉过短的小碎片。
    
    Args:
        skeleton: 骨架化图像 [H, W]
        min_length: 最小碎片长度（像素数）
        
    Returns:
        num_fragments: 碎片数
    """
    if skeleton.sum() == 0:
        return 0
    
    # 标记连通分量
    labeled_array, num_features = ndimage.label(skeleton)
    
    # 统计每个分量的长度
    fragment_lengths = []
    for i in range(1, num_features + 1):
        length = np.sum(labeled_array == i)
        if length >= min_length:
            fragment_lengths.append(length)
    
    return len(fragment_lengths)


def compute_topology_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    min_fragment_length: int = 10
) -> Dict[str, float]:
    """计算拓扑指标（CL-Break和Δβ₀）。
    
    所有计算均在ROI区域内进行。
    
    Args:
        pred: 预测结果 [H, W]
        target: 真实标签 [H, W]
        roi_mask: ROI掩码 [H, W]（可选）
        threshold: 二值化阈值
        min_fragment_length: 最小碎片长度
        
    Returns:
        metrics: 包含cl_break和delta_beta0的字典
        
    Note:
        - CL-Break: 预测骨架的碎片数（越低越好，理想接近GT的碎片数）
        - Δβ₀: |β₀(pred) - β₀(GT)|（连通分量数差异，越低越好）
    """
    try:
        # 二值化
        pred_binary = (pred > threshold).astype(np.uint8)
        target_binary = (target > 0).astype(np.uint8)
        
        # 应用ROI掩码
        if roi_mask is not None:
            pred_binary = pred_binary * (roi_mask > 0).astype(np.uint8)
            target_binary = target_binary * (roi_mask > 0).astype(np.uint8)
        
        # 骨架化
        pred_skeleton = skeletonize_vessel(pred_binary, roi_mask=None)  # 已应用ROI
        target_skeleton = skeletonize_vessel(target_binary, roi_mask=None)
        
        # CL-Break: 预测骨架碎片数
        pred_fragments = count_skeleton_fragments(pred_skeleton, min_fragment_length)
        
        # 计算GT的碎片数作为参考
        target_fragments = count_skeleton_fragments(target_skeleton, min_fragment_length)
        
        # CL-Break指标：预测的碎片数（数值越高说明断裂越严重）
        cl_break = float(pred_fragments)
        
        # Δβ₀: Betti数差异（使用过滤后的Betti0，与训练目标一致）
        pred_beta0 = compute_betti0_filtered(pred_binary, min_size=20)
        target_beta0 = compute_betti0_filtered(target_binary, min_size=20)
        delta_beta0 = float(abs(pred_beta0 - target_beta0))
        
        return {
            'cl_break': cl_break,
            'delta_beta0': delta_beta0,
            # 附加信息（用于调试）
            'pred_beta0': float(pred_beta0),
            'target_beta0': float(target_beta0),
            'pred_fragments': float(pred_fragments),
            'target_fragments': float(target_fragments)
        }
        
    except Exception as e:
        # 拓扑计算失败时返回默认值
        return {
            'cl_break': -1.0,
            'delta_beta0': -1.0,
            'pred_beta0': -1.0,
            'target_beta0': -1.0,
            'pred_fragments': -1.0,
            'target_fragments': -1.0,
            'error': str(e)
        }


def compute_all_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    compute_topology: bool = True
) -> MetricsResult:
    """计算所有指标（基础 + 拓扑）。
    
    Args:
        pred: 预测结果 [H, W]
        target: 真实标签 [H, W]
        roi_mask: ROI掩码 [H, W]（可选）
        threshold: 二值化阈值
        compute_topology: 是否计算拓扑指标
        
    Returns:
        result: MetricsResult对象
    """
    # 基础指标
    basic = compute_basic_metrics(pred, target, roi_mask, threshold)
    
    result = MetricsResult(
        dice=basic['dice'],
        iou=basic['iou'],
        precision=basic['precision'],
        recall=basic['recall']
    )
    
    # 拓扑指标
    if compute_topology:
        topo = compute_topology_metrics(pred, target, roi_mask, threshold)
        result.cl_break = topo['cl_break']
        result.delta_beta0 = topo['delta_beta0']
    
    return result


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """将PyTorch张量转换为NumPy数组。
    
    Args:
        tensor: PyTorch张量，形状[B, C, H, W]或[C, H, W]或[H, W]
        
    Returns:
        array: NumPy数组，形状[H, W]（如果是batch则取第一个样本）
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    array = tensor.detach().numpy()
    
    # 处理batch维度
    if array.ndim == 4:
        array = array[0, 0]  # [B, C, H, W] -> [H, W]
    elif array.ndim == 3:
        array = array[0]     # [C, H, W] -> [H, W]
    
    return array


class MetricsTracker:
    """指标追踪器，用于累积多batch的指标。
    
    示例：
        >>> tracker = MetricsTracker()
        >>> for batch in dataloader:
        ...     metrics = compute_all_metrics(pred, target, roi)
        ...     tracker.update(metrics)
        >>> avg_metrics = tracker.get_average()
    """
    
    def __init__(self):
        self.metrics_sum: Dict[str, float] = {}
        self.count: int = 0
    
    def update(self, metrics: MetricsResult) -> None:
        """更新指标累积。"""
        if not metrics.valid:
            return
        
        metrics_dict = {
            'dice': metrics.dice,
            'iou': metrics.iou,
            'precision': metrics.precision,
            'recall': metrics.recall,
            'cl_break': metrics.cl_break,
            'delta_beta0': metrics.delta_beta0
        }
        
        for key, value in metrics_dict.items():
            if key not in self.metrics_sum:
                self.metrics_sum[key] = 0.0
            self.metrics_sum[key] += value
        
        self.count += 1
    
    def get_average(self) -> Dict[str, float]:
        """获取平均指标。"""
        if self.count == 0:
            return {k: 0.0 for k in self.metrics_sum.keys()}
        
        return {k: v / self.count for k, v in self.metrics_sum.items()}
    
    def reset(self) -> None:
        """重置累积。"""
        self.metrics_sum.clear()
        self.count = 0


if __name__ == '__main__':
    # 简单测试
    print('测试指标计算...')
    
    # 创建模拟数据
    np.random.seed(42)
    pred = np.random.rand(512, 512)
    target = np.random.randint(0, 2, (512, 512)).astype(np.float32)
    roi = np.ones((512, 512), dtype=np.float32)
    roi[50:450, 50:450] = 1  # 模拟圆形ROI
    
    # 基础指标
    basic = compute_basic_metrics(pred, target, roi)
    print(f'基础指标: {basic}')
    
    # 拓扑指标
    topo = compute_topology_metrics(pred, target, roi)
    print(f'拓扑指标: CL-Break={topo["cl_break"]:.1f}, Δβ₀={topo["delta_beta0"]:.1f}')
    
    print('\n指标计算测试完成！')
