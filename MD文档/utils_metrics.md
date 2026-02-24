# utils_metrics.py 模块说明

## 功能概述

`utils_metrics.py`提供分割评估指标计算，包括：

- **基础指标**：Dice、IoU、Precision、Recall（均在ROI内计算）
- **拓扑指标**：
  - CL-Break（中心线碎片数）：骨架化后的连通分量数
  - Δβ₀（Betti误差）：预测与GT的连通分量数差异

## 接口定义（Input/Output）

### compute_basic_metrics函数

```python
def compute_basic_metrics(
    pred: np.ndarray,      # [H, W] 预测概率
    target: np.ndarray,    # [H, W] 真实标签
    roi_mask: Optional[np.ndarray] = None,  # ROI掩码
    threshold: float = 0.5
) -> Dict[str, float]     # {'dice': x, 'iou': x, 'precision': x, 'recall': x}
```

### compute_topology_metrics函数

```python
def compute_topology_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    min_fragment_length: int = 10
) -> Dict[str, float]     # {'cl_break': x, 'delta_beta0': x, ...}
```

### MetricsResult数据类

```python
@dataclass
class MetricsResult:
    dice: float = 0.0
    iou: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    cl_break: float = 0.0      # 中心线碎片数
    delta_beta0: float = 0.0   # Betti₀误差
```

## 关键概念说明

### CL-Break（Centerline Breakage）

**定义**：骨架化后的连通分量数

**意义**：
- 血管应该是连通的（理想情况下骨架是1个连通分量）
- 碎片数越多，说明血管断裂越严重
- 基线U-Net通常CL-Break > 50，加入拓扑约束后可降至<10

**计算流程**：
```
pred_binary -> skeletonize -> label -> count
```

### Δβ₀（Betti Number Error）

**定义**：|β₀(pred) - β₀(GT)|

**意义**：
- β₀表示二值图像的连通分量数
- Δβ₀=0表示预测与GT的连通性完全一致
- 反映整体拓扑结构差异

## 使用示例

```python
from utils_metrics import compute_all_metrics, tensor_to_numpy

# 假设pred和target是PyTorch张量
pred_np = tensor_to_numpy(pred)      # [H, W]
target_np = tensor_to_numpy(target)  # [H, W]
roi_np = tensor_to_numpy(roi)        # [H, W]

# 计算所有指标
result = compute_all_metrics(
    pred_np, target_np, roi_np,
    threshold=0.5,
    compute_topology=True
)

print(f'Dice: {result.dice:.4f}')
print(f'CL-Break: {result.cl_break:.1f}')
print(f'Δβ₀: {result.delta_beta0:.1f}')
```

## 依赖清单

**Python包**：
- `numpy`: 数组操作
- `torch`: 张量转换
- `scikit-image`: 骨架化（skimage.morphology.skeletonize）
- `scipy`: 连通区域标记（scipy.ndimage.label）

## ROI约束的重要性

**为什么必须在ROI内计算？**

DRIVE数据集的ROI是圆形眼底区域，黑色背景：
- 背景像素（ROI外）不应参与指标计算
- 如果在全图计算，大量背景真负例会inflate accuracy
- 正确的做法：`pred[roi_mask>0]` vs `target[roi_mask>0]`

## 更新日志

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v1.0 | 2026-02-07 | 初始版本，实现基础指标和拓扑指标 |
| v1.1 | 2026-02-07 | 修正Betti数计算，添加ROI掩码约束 |
