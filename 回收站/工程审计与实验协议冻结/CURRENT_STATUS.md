# 当前事实清单 (Current Status Audit)

## 生成时间
2026-02-25

## 1. 训练入口

| 脚本 | 用途 | 数据加载方式 |
|------|------|-------------|
| `train_baseline.py` | U-Net基线训练（纯Dice Loss） | `get_combined_loaders()` |
| `train_with_topology.py` | 拓扑正则训练（Dice + Topo） | `get_combined_loaders()` |

## 2. 默认配置 (config.yaml)

### 数据集配置
```yaml
data:
  use_kaggle_combined: true    # 使用Kaggle联合数据集
  root: ./DRIVE                # DRIVE模式时使用
  train_ids: [21-36]           # DRIVE训练集（16张）
  val_ids: [37-40]             # DRIVE验证集（4张）
  test_ids: [01-20]            # DRIVE测试集（20张）
  img_size: 512                # 统一512×512
  in_channels: 3               # RGB 3通道（已统一）
```

### 训练超参
```yaml
training:
  batch_size: 4
  num_workers: 4
  learning_rate: 1.0e-4
  max_epochs: 200
  patience: 20                 # 早停耐心值
  enable_early_stopping: false # 当前禁用早停
  device: cuda
  seed: 42
```

### 拓扑损失配置
```yaml
topology:
  target_beta0: 5              # 目标连通分量数
  max_death: 0.5               # filtration阈值
  loss_scale: 100.0            # 损失缩放因子
```

## 3. 数据加载详情

### KaggleCombinedDataset (data_combined.py)
- **来源**: data/combined/ 本地目录（DRIVE+HRF+CHASE+STARE）
- **训练集**: Training/ 目录 (73张)
- **验证集**: Test/ 目录 (20张)
- **预处理**:
  - Resize + Pad到512×512（保持长宽比）
  - 图像: RGB [3,H,W]，Normalize(mean=0.5, std=0.5)
  - 标签: [1,H,W]，二值化
  - ROI: 全1 mask [1,H,W]
- **数据增强**:
  - 水平翻转 (p=0.5)
  - 垂直翻转 (p=0.5)
  - 随机旋转 -15°~+15° (p=0.5)

### DRIVEDataset (data_drive.py)
- **训练**: 21-36号 (16张)
- **验证**: 37-40号 (4张)
- **预处理**: 同上，统一RGB 3通道

## 4. 拓扑损失形式 (topology_loss.py)

### 经典稳定版公式
```python
# 1. 子水平集filtration
filtration = 1.0 - prob_map    # 高prob -> 低filtration（早birth）

# 2. 计算0维持续同调
pd = cripser.compute_ph_torch(filtration, maxdim=0, filtration="V")

# 3. 提取lifetimes
lifetimes = deaths - births

# 4. 保留前target_beta0个最长组件
if len(lifetimes) > target_beta0:
    lifetimes = topk(lifetimes, target_beta0)

# 5. MSE到目标lifetime=0.5
loss = MSE(lifetimes, 0.5) * loss_scale
```

### 参数
- `target_beta0=5`: 保留5个最长持续组件
- `max_death=0.5`: filtration域最大death值
- `loss_scale=100.0`: 损失缩放（显示值已缩放）

## 5. 评估指标计算 (utils_metrics.py)

### Dice系数
```python
dice = 2 * |pred ∩ target| / (|pred| + |target|)
```

### CL-Break (连通性损失)
```python
# 计算预测和GT的0维Betti数差异
pred_components = label(pred_binary).max()
target_components = label(target_binary).max()
cl_break = |pred_components - target_components|
```

### Δβ₀
```python
delta_beta0 = pred_components - target_components
```

### 二值化阈值
- 默认: 0.5 (config.yaml: metrics.topology_threshold)
- 用于: CL-Break、Δβ₀计算

## 6. λ调度策略 (015策略)

```python
# 阶段1: 前30% epochs (0-60)
λ = 0.0

# 阶段2: 中间30% epochs (60-120)
λ = 0.0 → 0.1 (线性增长)

# 阶段3: 最后40% epochs (120-200)
λ = 0.1 → 0.5 (线性增长)
```

## 7. 日志输出

### CSV日志 (training_topo_log.csv)
```
epoch,train_loss,train_dice,train_loss_topo,val_dice,val_iou,val_precision,val_recall,cl_break,delta_beta0,lambda,lr
```

### 控制台输出
```
Epoch 1/200  (λ=0.000)
  Train Loss: 0.8234 | Train Dice: 0.5123 | Train Topo: 0.0839 (含loss_scale)
  Val Dice: 0.6789 | Val IoU: 0.5234 | Val Prec: 0.6123 | Val Rec: 0.7234
  CL-Break: 12.5 | Δβ₀: 8.2
```

## 8. 模型架构

- **Encoder**: ResNet34 (ImageNet预训练)
- **Input**: 3通道 RGB [B,3,512,512]
- **Output**: 1通道 logits [B,1,512,512]
- **参数量**: ~24.4M

## 9. 已知问题/注意事项

1. **早停当前禁用**: `enable_early_stopping: false`，会跑满200轮
2. **拓扑损失显示**: 日志中的"Train Topo"已是loss_scale后的值
3. **ROI处理**: Kaggle数据集使用全1 ROI（无FOV mask）
4. **验证集**: Kaggle模式使用官方Test/目录（20张），非DRIVE 37-40

