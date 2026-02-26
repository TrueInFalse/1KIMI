# 实验协议 (Experiment Protocol)

**版本**: v1.0  
**冻结日期**: 2026-02-25  
**适用范围**: 视网膜血管分割拓扑正则研究

---

## 1. 数据集来源与构成

### 1.1 Kaggle联合数据集（主线实验）

**来源**: Local path `data/combined/`  
**组成**: DRIVE + HRF + CHASE DB1 + STARE

| 子集 | 数量 | 来源目录 |
|------|------|---------|
| 训练集 | 73张 | `Training/images/` + `Training/masks/` |
| 验证集 | 20张 | `Test/images/` + `Test/masks/` |

**数据集切换**: 修改 `config.yaml`:
```yaml
data:
  use_kaggle_combined: true   # Kaggle联合模式
  # use_kaggle_combined: false  # 纯DRIVE模式
```

### 1.2 DRIVE数据集（锚点对照实验）

**来源**: `./DRIVE/`

| 子集 | ID范围 | 数量 |
|------|--------|------|
| 训练集 | 21-36 | 16张 |
| 验证集 | 37-40 | 4张 |
| 测试集 | 01-20 | 20张 |

---

## 2. Split规则

### 2.1 固定列表

**Kaggle联合数据集**: 使用官方提供的Training/Test划分，列表固定不变。

**DRIVE**: ID列表硬编码于 `config.yaml`:
```yaml
train_ids: [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
val_ids: [37, 38, 39, 40]
test_ids: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
```

### 2.2 随机种子

```yaml
training:
  seed: 42
```

**设置位置**: `train_baseline.py` 和 `train_with_topology.py` 的 `set_seed()` 函数。

---

## 3. 图像预处理

### 3.1 尺寸处理

```yaml
data:
  img_size: 512
```

**处理流程** (`data_combined.py`):
1. 保持长宽比缩放
2. Center Pad到512×512
3. 填充值: 0 (black)

### 3.2 通道与归一化

**输入**: RGB 3通道  
**归一化**: `T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])`

**注意**: 所有数据集（DRIVE原图是RGB）已统一为3通道输入。

### 3.3 ROI规则

- **Kaggle联合数据集**: 自动生成全1 ROI mask `[1, H, W]`
- **DRIVE**: 使用官方FOV mask（圆形视野）

### 3.4 数据增强（仅训练集）

| 增强类型 | 参数 | 概率 |
|---------|------|------|
| 水平翻转 | - | 0.5 |
| 垂直翻转 | - | 0.5 |
| 随机旋转 | -15° ~ +15° | 0.5 |

**旋转实现**: `TF.rotate(image, angle, fill=0)`，mask使用`NEAREST`插值保持二值性。

---

## 4. 训练超参

### 4.1 基础配置

```yaml
training:
  batch_size: 4
  num_workers: 4
  learning_rate: 1.0e-4
  max_epochs: 200
  device: cuda
```

### 4.2 优化器与学习率调度

- **优化器**: Adam
- **学习率调度**: CosineAnnealingLR
  - `T_max = max_epochs`
  - `eta_min = learning_rate * 0.01`

### 4.3 λ计划（015策略）

**总公式**: `L_total = L_Dice + λ * L_topo`

| 阶段 | Epoch范围 | λ值 | 说明 |
|------|-----------|-----|------|
| 1 | 0-59 | 0.0 | 纯Dice训练 |
| 2 | 60-119 | 0.0 → 0.1 | 轻度拓扑约束 |
| 3 | 120-199 | 0.1 → 0.5 | 加强拓扑约束 |

**实现**: `train_with_topology.py` 中的 `LambdaScheduler` 类。

### 4.4 早停策略

```yaml
training:
  enable_early_stopping: false  # 当前禁用
  patience: 20
```

**当前策略**: 跑满200轮，不启用早停（确保充分训练）。

---

## 5. 评估指标定义

### 5.1 Dice系数

```python
dice = 2 * |pred ∩ target| / (|pred| + |target| + 1e-7)
```

**阈值**: 0.5（二值化）

### 5.2 CL-Break (Connectivity Loss)

```python
from skimage.measure import label
pred_components = label(pred_binary, connectivity=1).max()
target_components = label(target_binary, connectivity=1).max()
cl_break = |pred_components - target_components|
```

### 5.3 Δβ₀

```python
delta_beta0 = pred_components - target_components
```

### 5.4 阈值策略（**冻结策略B**）

**策略**: 验证集扫描一次，选定最优阈值后冻结，测试集使用冻结阈值。

**当前冻结阈值**: 0.5（所有指标统一使用）

**理由**: 
- Dice对阈值不敏感（极值点附近平坦）
- CL-Break/Δβ₀在0.5处与视觉感知对齐
- 避免阈值调参刷指标嫌疑

---

## 6. 一键复现命令

### 6.1 基线模型 (Baseline)

```bash
python train_baseline.py
```

**输出**: `checkpoints/best_model.pth`  
**日志**: `logs/training_log.csv`

### 6.2 拓扑正则模型 (Topo)

```bash
python train_with_topology.py
```

**输出**: `checkpoints/best_model_topo.pth`  
**日志**: `logs/training_topo_log.csv`

### 6.3 评估脚本

```bash
python evaluate.py --checkpoint checkpoints/best_model_topo.pth --split val
```

---

## 7. 复现性保证

### 7.1 关键指标抖动范围

**同commit重复运行2次**，预期抖动:

| 指标 | 预期抖动范围 | 原因 |
|------|-------------|------|
| Val Dice | ±0.005 | 数据增强随机性 |
| CL-Break | ±1.0 | 二值化边界效应 |
| Δβ₀ | ±0.5 | 连通分量计数整数特性 |

### 7.2 硬件环境

- **GPU**: RTX 3080 10GB
- **CUDA**: 与PyTorch配套
- **PyTorch**: 2.8+

### 7.3 关键依赖版本

```
torch>=2.0.0
segmentation-models-pytorch>=0.3.0
cripser==0.0.25
scikit-image>=0.20.0
```

---

## 8. 协议变更记录

| 日期 | 版本 | 变更内容 |
|------|------|---------|
| 2026-02-25 | v1.0 | 初始冻结 |

---

**注意**: 任何协议变更需更新此文档版本号，并记录变更理由。
