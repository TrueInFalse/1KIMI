# Topo(λ=0) vs Baseline 差异调查报告

## 现象

20轮实验对比：
| 指标 | Topo(λ=0) | Baseline | 差距 |
|------|-----------|----------|------|
| Val Dice | **0.714** | 0.605 | **+0.109** |
| Val Precision | **0.657** | 0.462 | **+0.195** |
| Val Recall | 0.785 | **0.888** | -0.103 |

Topo(λ=0) 显著优于 Baseline，且表现为 Precision 高、Recall 低。

## 排查过程

### 1. 评估协议检查 ✅
- **Baseline 验证循环**: `train_baseline.py:290-348`
- **Topo 验证循环**: `train_with_topology.py:456-522`
- **结论**: 两者都使用 `compute_basic_metrics(pred, target, roi, threshold)`，ROI 应用逻辑相同

### 2. ROI 一致性检查 ✅
- **两个脚本**: 都使用 `get_combined_loaders(config)`
- **Config**: `kaggle_roi.mode = 'fov'`
- **验证**: `verify_roi_usage.py` 确认 roi_mean≈0.52（FOV模式）
- **结论**: ROI 来源一致

### 3. 关键发现：随机种子不一致 ⚠️ FIXED
- **Baseline**: `train_baseline.py:488-496` 定义 `set_seed()`，在 `main()` 中调用
- **Topo(原)**: 无种子设置
- **修复**: 添加 `set_seed()` 到 `train_with_topology.py`，与 Baseline 完全一致

### 4. 其他可能差异（待验证）

即使种子对齐，以下因素仍可能导致差异：

#### 4.1 代码路径差异
- **Baseline**: 纯 DiceLoss
- **Topo(λ=0)**: DiceLoss + λ*TopoLoss (λ=0，但 topo loss 计算代码仍会执行)

Topo loss 计算是否有副作用？
```python
# topology_loss.py:72
prob = prob * roi + (1.0 - roi) * 0.0  # 创建新张量，无副作用
```
✅ 无副作用，PyTorch 操作创建新张量。

#### 4.2 训练循环细节
- **Baseline**: `train_baseline.py:251-284`
- **Topo**: `train_with_topology.py:352-437`

两者都使用：
- 相同的 optimizer: `Adam(lr=1e-4)`
- 相同的 scheduler: `CosineAnnealingLR(T_max=max_epochs, eta_min=lr*0.01)`
- 相同的 loss: `smp.losses.DiceLoss(mode='binary', from_logits=True)`
- 相同的梯度裁剪: `clip_grad_norm_(max_norm=1.0)`

#### 4.3 数据加载顺序
即使种子相同，如果数据加载器初始化顺序不同，可能导致 shuffle 结果不同。

## 结论与建议

### 已确认的差异
| 因素 | 状态 | 影响 |
|------|------|------|
| 评估协议 (ROI/阈值/指标函数) | ✅ 一致 | 无 |
| 数据预处理 | ✅ 一致 | 无 |
| 随机种子 | ⚠️ 已修复 | 可能是主因 |
| 优化器/调度器 | ✅ 一致 | 无 |

### 下一步行动

**必须重新运行对照实验**（种子已对齐）来验证：

```bash
# Exp-1: Baseline（20轮）
python train_baseline.py

# Exp-2: Topo(λ=0)（20轮，种子已对齐）
python train_with_topology.py
```

**预期结果**：
- 如果种子是主因，两者 Dice 应该接近（差距 < 0.01）
- 如果仍有显著差距，需要进一步检查数据加载顺序或其他隐藏差异

### 如果仍有差异的排查方向

1. **数据加载顺序**: 在 `get_combined_loaders` 中打印 shuffle 后的第一个 batch
2. **验证集指标复算**: 保存同一 checkpoint，用两个脚本的验证函数分别计算
3. **梯度检查**: 比较两个脚本在某一步的梯度是否一致

## 交付物

1. ✅ 修复：添加随机种子设置到 `train_with_topology.py`
2. ✅ 验证：ROI 使用一致性确认
3. ✅ 验证：评估协议一致性确认
4. ⏳ 待完成：种子对齐后的对照实验
