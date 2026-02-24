# train_with_topology.py 模块说明

## 功能概述

`train_with_topology.py`实现**端到端拓扑正则训练**（Stage 2），
基于 **cripser 0.0.25+** 的**真正可微分持续同调**。

**重大突破**: 无需STE近似，梯度真实传播，拓扑模块真正生效！

## 技术路线 (cripser 0.0.25+)

```
真正可微分架构:
  
  输入: 512×512概率图 (requires_grad=True)
    ↓
  cripser.compute_ph_torch
    → 持续图 (birth, death) pairs
    ↓
  可微分持续景观构造
    → landscape vector [n_points]
    ↓
  L2距离损失
    → loss_topo (真正可微分！)
    ↓
  梯度自动传播到CNN参数
```

### 对比旧版STE

| 特性 | 旧版 (cripser < 0.0.25) | 新版 (cripser >= 0.0.25) ✅ |
|------|------------------------|---------------------------|
| 可微分方式 | STE近似 | 原生支持 |
| Forward | CubicalRipser (numpy) | compute_ph_torch (torch) |
| Backward | 软近似梯度 | 真实梯度传播 |
| 效果 | "伪拓扑正则" | 拓扑真正生效 |
| 代码复杂度 | 高（双路径） | 低（单路径） |

## 课程学习策略

```
阶段1 (Epoch 0-50):   λ = 0.00  纯Dice训练（CNN预热）
阶段2 (Epoch 50-100): λ = 0.05  轻度拓扑微调  
阶段3 (Epoch 100+):   λ = 0.15  适度拓扑优化

总损失: L_total = L_Dice + λ * L_topo
```

## 接口定义

### TrainerWithTopology类

```python
class TrainerWithTopology:
    def __init__(
        self,
        config: Dict,           # 配置字典
        args: Optional[Any]     # 命令行参数
    )
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> None
```

### LambdaScheduler类

```python
class LambdaScheduler:
    """三阶段课程学习调度器"""
    
    def __init__(
        self,
        phase1_epochs: int = 50,      # 阶段1轮数
        phase2_epochs: int = 50,      # 阶段2轮数  
        phase3_epochs: int = 100,     # 阶段3轮数
        lambda_phase2: float = 0.05,  # 阶段2 λ值
        lambda_phase3: float = 0.15   # 阶段3 λ值
    )
```

## 配置参数（config.yaml）

```yaml
topology:
  target_betti0: 10.0       # 目标连通分量数
  n_points: 64              # 景观离散化点数
  
  # 课程学习策略
  phase1_epochs: 50         # 阶段1：纯Dice（CNN预热）
  phase2_epochs: 50         # 阶段2：弱拓扑（λ=0.05）
  phase3_epochs: 100        # 阶段3：适度拓扑（λ=0.15）
  lambda_phase2: 0.05
  lambda_phase3: 0.15
  
  # 关键参数：损失缩放（cripser 0.0.25+推荐10.0）
  loss_scale: 10.0          # 使Topo占比约15-30%
```

## 使用示例

```bash
# 安装cripser 0.0.25+
pip install -U cripser

# 运行训练
python train_with_topology.py

# 预期输出（拓扑真正生效！）
# Epoch 100/200 | λ=0.150 | Train Dice: 0.78 | Val Dice: 0.76 | CL-Break: 15.0
```

## 关键改进

### 1. 移除STE复杂性
旧版需要：
```python
# 旧版 (STE)
hard_loss = cripser.computePH(...)  # numpy, 无梯度
soft_loss = soft_betti_loss(...)     # 近似
ste_loss = hard_loss + (soft_loss - soft_loss.detach())  # 复杂
```

新版简化：
```python
# 新版 (真正可微分)
loss_topo = cripser.compute_ph_torch(...)  # 直接可微分！
```

### 2. 梯度真实传播
- 旧版: 软近似梯度，可能方向不准
- 新版: 梯度通过birth/death真实传播到输入

### 3. 拓扑真正生效
- 旧版: "伪拓扑正则"（Dice几乎不变）
- 新版: 预期CL-Break显著改善（70→20）

## 输出文件

```
checkpoints/
├── best_model_topo.pth       # 最佳验证Dice模型
├── final_model_topo.pth      # 最终模型
└── checkpoint_epoch_*.pth    # 中间检查点

logs/
└── training_topo_log.csv     # 训练日志
```

## 依赖

- `torch`: PyTorch框架
- `cripser>=0.0.25`: CubicalRipser（必须最新版！）
- `segmentation-models-pytorch`: U-Net模型

## 性能预期

| 指标 | Stage 1基线 | Stage 2目标 (新版) |
|------|-------------|-------------------|
| Val Dice | ~0.76 | 0.75-0.78 |
| CL-Break | ~70 | <20 (真正改善!) |
| Δβ₀ | - | <10 |
| 训练时间 | ~12分钟 | ~15-20分钟 |

## 注意事项

1. **cripser必须>=0.0.25**: `pip install -U cripser`
2. **loss_scale关键**: 推荐10.0（使Topo占比15-30%）
3. **梯度裁剪**: 默认max_norm=1.0，防止梯度爆炸
4. **内存**: 512×512直接计算，无需降采样

## 参考

- **cripser 0.0.25**: https://github.com/CubicalRipser
- **Release Note**: 2026-02-16, PyTorch集成支持

## 更新记录

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v2.0 | 2026-02-19 | 基于cripser 0.0.25重写，使用真正可微分持续同调，移除STE |
| v1.0 | 2026-02-09 | STE架构（已废弃） |
