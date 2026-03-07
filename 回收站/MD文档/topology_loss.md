# topology_loss.py 模块说明

## 功能概述

`topology_loss.py`实现基于**cripser 0.0.25+**的**真正可微分持续同调损失**。

**重大突破**: cripser 0.0.25 (2026-02-16发布) 新增 `compute_ph_torch` 函数，支持：
- **真正可微分**: 梯度通过birth/death值传播到输入
- **PyTorch原生**: 输入输出都是torch.Tensor
- **无需STE**: 不再需要Straight-Through Estimator近似

## 技术路线对比

### 旧版 (cripser < 0.0.25)
```
STE架构:
  Forward: CubicalRipser (numpy, 无梯度)
  Backward: 软近似 (SoftBetti/PersistenceLandscape)
  问题: 软梯度方向可能不准确，"伪拓扑正则"
```

### 新版 (cripser >= 0.0.25) ✅
```
真正可微分:
  Forward: cripser.compute_ph_torch (torch, 有梯度!)
  Backward: 梯度真实传播
  优势: 拓扑模块真正影响CNN参数
```

## 接口定义

### CubicalRipserLoss类（主类）

```python
class CubicalRipserLoss(nn.Module):
    def __init__(
        self,
        target_betti0: float = 10.0,    # 目标连通分量数
        n_points: int = 64,              # 景观离散化点数
        filtration: str = 'V',           # 过滤类型
        loss_scale: float = 10.0,        # 损失缩放因子（推荐10.0）
        edge_protect: bool = False       # 边缘保护
    )
    
    def forward(
        self,
        pred: torch.Tensor,              # [B, 1, H, W] 概率图
        roi_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor                    # 标量损失（真正可微分！）
```

### 向后兼容别名

```python
SoftPersistenceLandscape = CubicalRipserLoss  # v2.0别名
SoftBettiLoss = CubicalRipserLoss             # v1.0别名
```

## 计算流程

```
输入: pred [B, 1, H, W] (概率图，requires_grad=True)
  ↓
cripser.compute_ph_torch(..., maxdim=0)
  → 持续图: [N, 9] tensor (dim, birth, death, ...)
  ↓
提取0维对: birth [M], death [M]
  ↓
构造持续景观（可微分）:
  Λ(x) = max_i { min(x-birth_i, death_i-x) }
  （使用softmax近似max，softmin近似min）
  ↓
L2距离: loss = MSE(landscape, target_landscape)
  ↓
梯度自动传播到pred!
```

## 关键特性

### 1. 真正可微分
```python
pred = torch.rand(2, 1, 512, 512, requires_grad=True)
loss_fn = CubicalRipserLoss(target_betti0=10.0, loss_scale=10.0)

loss = loss_fn(pred)  # 前向
loss.backward()       # 反向 - 梯度真实传播！

print(pred.grad is not None)  # True
print(pred.grad.abs().sum() > 0)  # True
```

### 2. 拓扑占比可控
推荐`loss_scale=10.0`，使拓扑损失占总损失的15-30%：
```
L_total = L_Dice + λ * L_topo

示例:
  Dice Loss: 0.50
  Topo Loss: 2.50 (with loss_scale=10.0)
  λ=0.15: λ*Topo = 0.375
  Topo占比: 0.375 / (0.50+0.375) ≈ 27%
```

### 3. 无需降采样
旧版需要将512×512降采样到128×128（cripser限制），
新版可以直接处理512×512（但128×128更快）。

## 配置参数（config.yaml）

```yaml
topology:
  target_betti0: 10.0       # 目标连通分量数
  n_points: 64              # 景观离散化点数
  loss_scale: 10.0          # 关键！推荐10.0 (Topo占比15-30%)
  edge_protect: false       # 边缘保护（可选）
```

## 使用示例

```python
from topology_loss import CubicalRipserLoss

# 创建损失函数
loss_fn = CubicalRipserLoss(
    target_betti0=10.0,
    loss_scale=10.0  # 关键参数！
)

# 前向传播
pred = model(image)  # [B, 1, H, W], requires_grad=True
loss_topo = loss_fn(pred)

# 反向传播 - 梯度真实传播！
loss_topo.backward()
```

## 性能对比

| 指标 | 旧版 (STE) | 新版 (真正可微分) |
|------|-----------|------------------|
| 梯度真实性 | 近似（软启发式） | 真实（cripser原生） |
| Dice提升 | ~0 (伪拓扑) | 待验证 (真拓扑) |
| CL-Break改善 | 有限 | 预期显著改善 |
| 训练时间 | ~14分钟 | ~15-20分钟 |

## 依赖

```bash
pip install -U cripser>=0.0.25
```

## 参考

- **cripser 0.0.25**: https://github.com/CubicalRipser
- **PyTorch PH**: `cripser.compute_ph_torch`

## 更新记录

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v3.0 | 2026-02-19 | 基于cripser 0.0.25重写，使用真正可微分持续同调，移除STE |
| v2.0 | 2026-02-18 | STE架构（已废弃） |
| v1.0 | 2026-02-07 | 软Betti数（已废弃） |
