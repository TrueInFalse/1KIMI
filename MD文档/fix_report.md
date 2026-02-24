# 拓扑正则化修复报告

## 关键缺陷修正对照表

| 缺陷 | 原代码行号/问题 | 修正逻辑 |
|------|----------------|----------|
| **A** | topology_loss.py: 未使用1.0-prob | `filtration = 1.0 - prob`（子水平集，前景早birth） |
| **B** | topology_loss.py: 未处理4D输入 | `prob_map.squeeze(1)`确保2D [H,W] |
| **C** | topology_loss.py: 使用了.double()和numpy转换 | 移除.double()，全程保持torch.Tensor，直接传compute_ph_torch |
| **D** | topology_loss.py: _compute_landscape_differentiable软近似 | 删除该函数，改用`cripser.finite_lifetimes` + `torch.relu(deficit)` |
| **E** | train_with_topology.py: LambdaScheduler三阶段 | 改为`warmup_epochs=30`固定0.1 + `ramp_epochs=70`线性增至0.5 |
| **F** | topology_loss.py: 未处理ROI | `prob * roi + (1-roi)*0.0`，背景强制prob=0 |

## 关键代码变更

### topology_loss.py（删除~80行，新增TopologicalRegularizer类）

**删除**：
- `_compute_landscape_differentiable` 函数（软近似landscape）
- `target_landscape` 构造（人工目标）
- `.double()` 转换（破坏梯度图）
- `.cpu().numpy()` 任何转换

**新增核心逻辑**：
```python
# 子水平集filtration（缺陷A修复）
filtration = 1.0 - prob

# 纯2D输入（缺陷B修复）
if filtration.dim() > 2: filtration = filtration.squeeze()

# 原生torch调用（缺陷C修复）
pd = cripser.compute_ph_torch(filtration, maxdim=0, filtration="V")

# 直接使用finite持续对（缺陷D修复）
births, deaths = pd[dim0_mask, 1], pd[dim0_mask, 2]
finite_mask = torch.isfinite(deaths)
lifetimes = deaths_finite - births_finite
deficit = torch.relu(max_death - lifetimes)
```

### train_with_topology.py（4处修改）

**Line 44-45**: 导入保持不变

**Line 48-78**: LambdaScheduler重写（缺陷E修复）
```python
# 原：三阶段0→0.05→0.15
# 新：warmup=30固定0.1，ramp=70线性增至0.5
if epoch < 30: return 0.1
elif epoch < 100: return 0.1 + (epoch-30)/70 * 0.4
else: return 0.5
```

**Line 148-158**: 损失初始化改为TopologicalRegularizer

**Line 215-217**: 传入epoch参数，ROI转float

## 验证结果

运行 `python test_topo_fix.py`：

- **测试A**: Filtration方向 → 1-2个长bar（中心高prob早birth）
- **测试B**: 梯度流通 → `input.grad.norm() > 1e-6` 确认
- **测试C**: λ调度 → 0→0.1, 30→0.1, 40≈0.16, 70≈0.33, 100→0.5
- **测试D**: 5 epoch → Topo loss逐步下降

## 预期性能

- **Val Dice**: 0.75 → 0.80+（20-30 epoch内）
- **CL-Break**: 20+ → <14（降低≥30%）
- **Topo Loss**: ~0.3 → ~0.05
