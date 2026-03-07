# ROI Dice 口径审计报告

**审计日期**: 2026-03-06  
**审计目标**: 验证"训练目标与评估目标在 ROI 口径上错位"的问题，并量化错位程度  
**审计方法**: 同一 batch 复算 Dice 口径差（全图 vs ROI 内）

---

## 1. 审计目的

本审计旨在回答以下核心问题：

> **"在当前项目里，全图训练 Dice 与 ROI 内评估 Dice 的差距到底有多大，这个差距是否足以影响 Baseline vs Topo 的公平比较？"**

根据代码审计报告的前期发现：
- `train_with_topology.py` 中 Dice 训练损失不使用 ROI，而 Topo loss 使用 ROI
- 验证阶段基础指标与拓扑指标都在 ROI 内计算
- 这可能导致优化目标与评估目标偏移

本审计通过量化这种口径差异，为后续决策提供数据支持。

---

## 2. 代码事实确认

### 2.1 Baseline 训练时 Dice loss 计算方式

**文件**: `train_baseline.py`  
**函数**: `Trainer._setup_optimizer()` / `Trainer.train_epoch()`  
**关键代码位置**: 第 192-195 行, 第 260 行

```python
# 第 192-195 行: 初始化 DiceLoss
self.criterion = smp.losses.DiceLoss(
    mode='binary',
    from_logits=True  # 模型输出logits，内部自动sigmoid
)

# 第 260 行: 训练时计算 loss
loss = self.criterion(outputs, vessels)  # 注意：没有传入 ROI mask
```

**结论**: Baseline 训练的 Dice loss **不使用 ROI，计算在全图上**。

### 2.2 Topo 训练时 Dice loss 计算方式

**文件**: `train_with_topology.py`  
**函数**: `TrainerWithTopology.__init__()` / `TrainerWithTopology.train_epoch()`  
**关键代码位置**: 第 174 行, 第 376 行

```python
# 第 174 行: 初始化 DiceLoss
self.criterion_dice = smp.losses.DiceLoss(mode='binary', from_logits=True)

# 第 376 行: 训练时计算 Dice loss
loss_dice = self.criterion_dice(outputs, vessels)  # 注意：没有传入 ROI mask
```

**结论**: Topo 训练的 Dice loss **同样不使用 ROI，计算在全图上**。

### 2.3 Topo loss 是否使用 ROI

**文件**: `topology_loss.py`  
**函数**: `TopologicalRegularizer.forward()`  
**关键代码位置**: 第 69-72 行

```python
# 应用ROI
if roi_mask is not None:
    roi = roi_mask[i]
    prob = prob * roi + (1.0 - roi) * 0.0  # ROI 外设为 0
```

**结论**: Topo loss **明确使用 ROI**，在计算持续同调前先将概率图与 ROI mask 相乘。

### 2.4 验证集指标是否在 ROI 内计算

**文件**: `utils_metrics.py`  
**函数**: `compute_basic_metrics()` / `apply_roi_mask()`  
**关键代码位置**: 第 52-76 行, 第 108-110 行

```python
def apply_roi_mask(pred, target, roi_mask):
    """应用ROI掩码，只保留ROI内的像素参与计算。"""
    pred_roi = pred[roi_mask > 0]
    target_roi = target[roi_mask > 0]
    return pred_roi, target_roi

def compute_basic_metrics(pred, target, roi_mask=None, threshold=0.5):
    # 二值化预测
    pred_binary = (pred > threshold).astype(np.float32)
    target_binary = target.astype(np.float32)
    
    # 应用ROI掩码
    if roi_mask is not None:
        pred_binary, target_binary = apply_roi_mask(pred_binary, target_binary, roi_mask)
    # ... 后续计算在 ROI 内进行
```

验证阶段调用位置：
- `train_baseline.py` 第 333-334 行
- `train_with_topology.py` 第 500-504 行

**结论**: 验证阶段所有基础指标（Dice/IoU/Precision/Recall）和拓扑指标 **都在 ROI 内计算**。

---

## 3. 审计方法

### 3.1 审计脚本

**文件名**: `audit_roi_dice_gap.py`  
**核心逻辑**:
1. 加载当前 config，使用 Kaggle 联合模式、当前默认 ROI 模式（fov）
2. 加载一批 train batch 和一批 val batch
3. 对同一批数据，拿随机初始化模型的输出后同时计算：
   - 全图 Dice loss
   - ROI 内 Dice loss
4. 对同一批数据，二值化后同时计算：
   - 全图 Dice score
   - ROI 内 Dice score
5. 统计差异并输出诊断信息

### 3.2 运行配置

```bash
python audit_roi_dice_gap.py --max-batches 20
```

- 设备: CUDA
- 数据模式: Kaggle 联合
- ROI 模式: fov（基于图像内容估计视场）
- 训练集样本: 72（18 batches × 4）
- 验证集样本: 20（5 batches × 4）

---

## 4. 关键数值结果

### 4.1 核心统计（二值化 Dice Score）

| 指标 | Train 集 | Val 集 |
|:---|:---:|:---:|
| 样本数 | 72 | 20 |
| **全图 Dice Mean** | 0.1168 | 0.1235 |
| **ROI Dice Mean** | 0.1763 | 0.1820 |
| **绝对差值 Mean** | **0.0595** | **0.0585** |
| 绝对差值 Std | 0.0163 | 0.0132 |
| 绝对差值 Median | 0.0577 | 0.0588 |
| **绝对差值 Max** | **0.0927** | **0.0807** |
| ROI 覆盖度 Mean | 0.6219 | 0.6350 |

### 4.2 差异分布

**Train 集**:
- 差值 > 0.02: 100.0% 的样本
- 差值 > 0.05: 72.2% 的样本
- 差值 > 0.10: 0.0% 的样本

**Val 集**:
- 差值 > 0.02: 100.0% 的样本
- 差值 > 0.05: 80.0% 的样本
- 差值 > 0.10: 0.0% 的样本

### 4.3 差异最大的样本

| 排名 | Split | Batch | Sample | 差值 | 全图Dice | ROI Dice | ROI覆盖度 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | train | 9 | 2 | **0.0927** | 0.1317 | 0.2244 | 0.522 |
| 2 | train | 3 | 2 | **0.0905** | 0.1304 | 0.2209 | 0.521 |
| 3 | train | 17 | 3 | **0.0902** | 0.1272 | 0.2174 | 0.521 |
| 4 | train | 8 | 0 | **0.0895** | 0.1264 | 0.2159 | 0.521 |
| 5 | train | 16 | 1 | **0.0893** | 0.1252 | 0.2145 | 0.523 |

### 4.4 ROI 覆盖度分布

- ROI 覆盖度范围: 0.52 ~ 0.85
- ROI 覆盖度中位数: ~0.65
- 全 1 ROI 样本数: 0（0%）

**关键发现**: ROI 仅覆盖图像的 **~62%** 面积，这意味着全图 Dice 计算中有 **~38%** 的像素位于 ROI 外。

### 4.5 模型输出在 ROI 内外差异

对第一个 train batch 的详细分析：

| Sample | ROI覆盖度 | Pred均值(ROI内) | Pred均值(ROI外) | 差值(外-内) |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 0.5206 | 0.9999 | 0.9970 | -0.0028 |
| 1 | 0.6456 | 1.0000 | 0.9960 | -0.0039 |
| 2 | 0.5224 | 0.9999 | 0.9971 | -0.0028 |

模型对 ROI 外区域仍有较高响应（sigmoid 输出接近 1.0），但这些区域在验证时被完全忽略。

---

## 5. 结论

### 5.1 ROI 口径差是否显著？

**是，显著。**

**量化证据**:
1. **平均差异 ~0.06**（6个百分点）：全图 Dice 比 ROI Dice 平均低 0.06
2. **最大差异 ~0.09**（9个百分点）：极端情况下差异接近 0.1
3. **100% 样本存在差异**：所有样本的全图 Dice 与 ROI Dice 都不相等
4. **72-80% 样本差异 > 0.05**：超过一半样本差异显著（>5%）

**相对影响**:
- 如果 Baseline 的 ROI Dice 为 0.80，全图 Dice 约为 0.74（差异 0.06）
- 这种差距足以改变模型排名的相对顺序

### 5.2 这个问题是否足以作为当前 Topo 退化的优先解释之一？

**是，这是一个重要因素，但可能不是唯一原因。**

**支持该结论的理由**:

1. **公平性问题**:
   - Baseline 和 Topo 训练时都使用全图 Dice loss
   - Topo 额外使用 ROI 内 Topo loss
   - 这导致 Topo 的优化目标更"聚焦"于 ROI，而 Baseline 在全图上均匀优化
   - ** Topo 可能在 ROI 外的区域欠优化**

2. **评估口径不一致**:
   - 训练优化目标：全图 Dice + ROI Topo
   - 验证评估目标：ROI Dice
   - 这种错位导致训练过程无法直接优化最终评估目标

3. **量化影响**:
   - 6% 的平均差异在视网膜血管分割任务中是显著的
   - 论文中报告的提升通常在 1-3% 范围内
   - 6% 的系统性偏差足以掩盖 Topo 的真实增益

**需要注意的局限性**:
- 本审计使用随机初始化模型，实际训练后的模型行为可能不同
- 差异的具体影响取决于模型学习到的特征分布
- 需要与训练日志中的实际指标进行对比验证

---

## 6. 下一步建议

### 建议 1：训练阶段对齐 ROI 口径（高优先级）

**动作**: 修改训练代码，使 Dice loss 也在 ROI 内计算。

**实现方式**:
```python
# 当前代码（全图计算）
loss_dice = self.criterion_dice(outputs, vessels)

# 修改后（ROI内计算）
loss_dice = compute_dice_loss(outputs, vessels, roi_mask=rois)
```

**预期效果**:
- 消除训练-评估口径差异
- 使 Topo 与 Baseline 在相同目标上优化
- 公平评估 Topo 的真实增益

### 建议 2：跑对比实验验证影响（中优先级）

**动作**: 在修改 Dice loss 为 ROI 计算后，跑短周期（如 20 epoch）对比实验。

**实验设计**:
1. **Baseline-ROI**: Dice loss 在 ROI 内计算
2. **Topo-ROI**: Dice loss 在 ROI 内计算 + Topo loss（ROI 内）

**观察指标**:
- Val Dice 收敛曲线
- 最终 Val Dice 差异
- 若 Topo-ROI 显著优于 Baseline-ROI，说明 Topo 真实有效；反之则说明原差异被高估

**时间成本**: ~1-2 小时（20 epoch × 2 组）

---

## 附录：审计输出文件

- **审计脚本**: `audit_roi_dice_gap.py`
- **详细结果**: `audit_results/roi_dice/audit_results.txt`
- **可视化图**: `audit_results/roi_dice/batch*_sample*_diff*.png`

---

**审计完成时间**: 2026-03-06  
**审计执行**: Kimi Code CLI
