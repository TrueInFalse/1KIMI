# Topo(λ=0) vs Baseline 差异分析报告

## 日志对比关键发现

| 指标 | Topo(λ=0) E20 | Baseline E20 | 差距 |
|------|---------------|--------------|------|
| Val Dice | 0.7142 | 0.6053 | **+0.1089** |
| Val IoU | 0.5592 | 0.4387 | +0.1205 |
| Val Precision | 0.6569 | 0.4619 | **+0.1950** |
| Val Recall | 0.7852 | 0.8881 | -0.1029 |
| CL-Break | 11.7 | 34.2 | -22.5 |
| Δβ₀ | 17.7 | 13.2 | +4.5 |

**关键观察**：
- Topo Precision 显著高于 Baseline (+0.195)
- Topo Recall 低于 Baseline (-0.103)
- 这表明 Baseline 可能有更多假阳性（预测为前景但实际是背景）

## 可能原因分析

### 1. 随机种子不一致（已修复）
- **Baseline**: `set_seed(42)` 在 main 中调用
- **Topo(原)**: 无种子设置
- **状态**: ✅ 已添加种子设置到 Topo

### 2. 评估协议检查
- **Baseline 验证循环**: 使用 `compute_basic_metrics(pred, target, roi, threshold)`
- **Topo 验证循环**: 使用相同函数
- **ROI 来源**: 两个脚本都从 data_combined 获取 ROI
- **状态**: 需要验证两个脚本实际使用的 ROI 是否相同

### 3. 数据加载器检查
- **两个脚本**: 都使用 `get_combined_loaders(config)`
- **ROI 模式**: 默认 'fov'
- **状态**: 需要验证实际加载的 ROI

### 4. 训练过程差异（即使 λ=0）
- **Baseline**: 纯 DiceLoss
- **Topo(λ=0)**: DiceLoss + λ*TopoLoss (λ=0, 但代码路径存在)
- **关键问题**: Topo 脚本是否在计算 topo_loss 时有副作用（如 ROI 外置零影响 pred）？

## 验证实验设计

### Exp-1: 验证 ROI 一致性
在验证时打印每个样本的 ROI 统计，对比两个脚本。

### Exp-2: 同一模型输出，两套评估
保存模型输出，分别用两个脚本的评估函数计算指标。

### Exp-3: 完整对照（种子对齐后）
重新跑 20 轮，对比 Topo(λ=0) vs Baseline。

## 下一步行动
1. 运行对照实验（种子已对齐）
2. 验证 ROI 实际使用情况
3. 如果仍有差异，检查 topo_loss 计算是否有副作用
