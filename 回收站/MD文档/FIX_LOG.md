# FIX LOG

本文件承载从 README 拆分出的历史修复记录与版本演进说明。

## 更新记录

- **v1.3** (2026-02-09):
  - 解决过连通问题（Over-Connecting）
  - 课程学习重构：0-50ep纯Dice，50-100ep弱拓扑(λ=0.05)，100+ep适度拓扑(λ=0.15)
  - 降低lambda_end从0.5到0.15，防止拓扑损失过度主导
  - 放松target_betti0从5到10，允许细枝末节合理断裂
  - 添加边缘保护机制：高置信度前景(>0.9)不降低，背景(<0.1)不提高
  - 修复Precision暴跌和假阳性桥接问题

- **v1.2** (2026-02-09): 
  - 实现端到端拓扑正则训练（STE + CubicalRipser）
  - 双分辨率策略：512×512 → 128×128
  - λ预热：0-30ep=0.1, 30-100ep→0.5
  - 目标Betti0=5（过滤后的金标准）
  - 完整的指标日志输出（含CL-Break、Δβ₀、时间）
  - 修复cripser输入格式（float64）
  - 修复验证指标计算（带min_size过滤）
  - 删除静默异常捕获，暴露真实错误

- **v1.1** (2026-02-07): 
  - 添加本地ResNet34权重加载（绕开网络请求）
  - 增强数据增强策略（翻转、旋转、亮度、对比度）
  - 改进CL-Break计算（添加形态学开运算去除噪声）
  - 添加训练曲线图自动生成
  - 优化训练日志输出格式

- **v1.0** (2026-02-07): 初始版本，Stage 0-1完成
  - 数据验证框架
  - U-Net基线训练
  - 拓扑指标计算
  - 可视化模块

## 依赖关系

```
train_baseline.py
├── config.yaml
├── data_drive.py
├── model_unet.py
└── utils_metrics.py

evaluate.py
├── config.yaml
├── data_drive.py
├── model_unet.py
└── utils_metrics.py

visualize_results.py
├── config.yaml
├── data_drive.py
└── model_unet.py
```

## 注意事项

1. **权重加载**：首次运行会自动下载ResNet34 ImageNet预训练权重到本地，后续从本地加载
2. **GPU内存**：512×512图像+ResNet34需要约6GB显存
3. **训练时间**：每轮约5-10秒（RTX 3080），总时间约15-30分钟

## 作者

AI Assistant

## 许可证

仅供研究使用

---

## 修复记录 (Fix Log)



### 2026-02-24: 人工修复说明（新增git版本控制）

**问题描述**：混合模式出现严重问题，进行版本回退，回退到kaggle联合数据集训练、验证模式，并修复若干bug。

**修复列表**:
  - 运行结果：
  ```
  Epoch 200/200  (λ=0.500)
  Train Loss: 0.1936 | Train Dice: 0.8240 | Train Topo: 0.0026
  Val Dice: 0.7751 | Val IoU: 0.6347 | Val Prec: 0.8028 | Val Rec: 0.7512
  CL-Break: 6.6 | Δβ₀: 14.0
  Each Time: 22s | Total Time: 1h 12m | ETA: 0s | LR: 0.000001
  \n================================================================================
  训练完成
  开始时间: 2026-02-24 11:59:51
  结束时间: 2026-02-24 13:11:37
  总用时: 71m 46s
  平均每轮: 22s
  最佳验证Dice: 0.7766
  最佳模型: checkpoints/best_model_topo.pth
  ```
  - 新增git版本控制



### 2026-02-24: 混合训练模式修复

**问题描述**: 实现 Kaggle Combined (73张) 训练 + DRIVE 37-40 (4张) 验证的混合模式时遇到多个bug。

**修复列表**:

#### 1. 拓扑损失参数不匹配
**问题**: `train_with_topology.py` 使用旧参数 `target_beta0=12`, `max_death=0.5`，但新版 `TopologicalRegularizer` 需要 `loss_scale`
**修复**: 
- 更新 `config.yaml`: `target_beta0: 4.0`, `loss_scale: 300.0`
- 修改 `train_with_topology.py` 从config读取参数

#### 2. `get_drive_val_loader()` 参数错误
**问题**: `data_combined.py` 中使用 `is_train=False`, `roi_margin=20`，但 `DRIVEDataset` 实际参数是 `is_training`, 无 `roi_margin`
**修复**: 
```python
# data_combined.py
val_dataset = DRIVEDataset(
    root=data_cfg['root'],
    image_ids=[37, 38, 39, 40],
    img_size=data_cfg['img_size'],
    in_channels=1,  # 添加
    is_training=False  # 修正参数名
)
```

#### 3. 通道不匹配 (模型3通道 vs 验证1通道)
**问题**: Kaggle模式模型输入为3通道(RGB)，但DRIVE验证集是1通道(灰度)
**修复**: 
- `train_with_topology.py`: 保存 `self.in_channels` 属性
- `validate()` 方法中添加通道适配:
```python
if images.shape[1] == 1 and self.in_channels == 3:
    images = images.repeat(1, 3, 1, 1)  # [B,1,H,W] -> [B,3,H,W]
```

**当前状态**: 混合模式训练正常运行，使用DRIVE 37-40作为验证集进行考核。

---


### 2026-02-24: 拓扑损失数值爆炸修复

**问题**: 训练日志显示 `Train Topo: 83847728184`（83亿），数值爆炸。

**根因**: 
- 训练初期网络随机，产生~500个连通分量
- 惩罚公式 `(num_bars - 4)^2 * 300` → 496² * 300 = 7300万+

**修复** (`topology_loss.py`, `config.yaml`):
```python
# 限制excess惩罚上限，防止初期爆炸
excess_penalty = min(excess, 50.0)  # 添加上限
loss_scale = 10.0                   # 从300降至10
```

**修复后**: `Train Topo: 9543`（正常范围）


### 2026-02-24: 回退到稳定Kaggle联合模式

**问题**: 混合训练模式（Kaggle训练 + DRIVE验证）引入过多复杂性：
- 通道不匹配（Kaggle RGB 3ch vs DRIVE Gray 1ch）
- 拓扑损失数值不稳定
- 验证逻辑复杂

**回退修改**:

| 文件 | 修改 |
|------|------|
| `data_combined.py` | `_get_mixed_kaggle_drive_loaders()` 改为使用Kaggle Test目录作为验证集（20张），而非强制DRIVE 37-40 |
| `train_with_topology.py` | 移除 `get_drive_val_loader()` 调用和通道适配代码 |
| `config.yaml` | 简化配置 |

**回退后状态**:
- 训练集：73张（Kaggle Training/）
- 验证集：20张（Kaggle Test/）
- 通道：RGB 3通道（一致）
- 目标：Val Dice >= 0.80


### 2026-02-24: 拓扑损失参数稳定化

**问题**: 拓扑损失数值不稳定，需要固定参数避免爆炸。

**修改**:
- `topology_loss.py`: 固定 `target_beta0=5`, `loss_scale=10.0`
- `config.yaml`: 更新拓扑损失参数
- `train_with_topology.py`: 删除自动计算Betti数代码

**参数说明**:
- target_beta0=5: 经验值，适合视网膜血管树结构（通常4-6个主要连通区域）
- loss_scale=10.0: 历史300.0导致爆炸，10.0更稳定


### 2026-02-24: 回归经典稳定拓扑损失

**问题**: 新公式（excess+short）复杂且不稳定。

**修改**:
- `topology_loss.py`: 回归经典公式 - 保留前target_beta0个最长组件 + MSE到0.5
- `config.yaml`: 简化拓扑参数，只保留target_beta0和max_death
- `train_with_topology.py`: 简化初始化代码

**经典公式**:
```python
# 保留前target_beta0个最长持续组件
if len(lifetimes) > target_beta0:
    lifetimes = topk(lifetimes, target_beta0)

# MSE到目标lifetime=0.5
loss = MSE(lifetimes, 0.5)
```

**优势**: 简单稳定，无数值爆炸风险。


### 2026-02-24: 添加loss_scale参数和早停统一开关

**修改1: loss_scale参数**
- `topology_loss.py`: 添加 `loss_scale` 参数（默认1.0）
- `config.yaml`: 添加 `topology.loss_scale: 1.0`
- `train_with_topology.py`: 传递loss_scale参数

**实际损失值计算**（用于参考调整）:
```
Dice损失: ~0.88
loss_scale=1.0: Topo=0.053, Ratio=6%
loss_scale=5.0: Topo=0.27,  Ratio=30%
loss_scale=10.0: Topo=0.53, Ratio=60%
```

**修改2: 早停统一开关**
- `config.yaml`: 添加 `training.enable_early_stopping: true`
- `train_baseline.py`: 根据开关启用/禁用早停
- `train_with_topology.py`: 根据开关启用/禁用早停

**使用方式**:
```yaml
# 启用早停（默认）
training:
  enable_early_stopping: true
  patience: 20

# 禁用早停（跑满max_epochs）
training:
  enable_early_stopping: false
```


### 2026-02-25: 实现015策略（λ课程学习新策略）

**策略说明（015策略）**:
- **阶段1（前30% epochs）**: λ=0，纯Dice训练，建立基础像素精度
- **阶段2（中间30% epochs）**: λ从0线性增长到0.1，轻度引入拓扑约束
- **阶段3（最后40% epochs）**: λ从0.1线性增长到0.5，加强拓扑优化

**以200 epochs为例**:
```
Epoch 0-60:   λ=0.0    (纯Dice)
Epoch 60-120: λ=0→0.1  (拓扑轻约束)
Epoch 120-200: λ=0.1→0.5 (拓扑强约束)
```

**设计理由**:
1. 前30%纯Dice：让网络先学习基础分割能力，避免早期拓扑干扰
2. 中间30%轻度拓扑：逐步引入连通性概念，平滑过渡
3. 最后40%加强拓扑：在后期精细化阶段重点优化血管连通性

**拓扑损失显示说明**:
- 日志中"Train Topo"值为**loss_scale后的值**（已乘以loss_scale）
- 当前config: loss_scale=100.0
- 如需调整拓扑权重，修改 `config.yaml` 中的 `topology.loss_scale`

## 2026-03-01: 止血调参完成 - Topo Loss 不降低 Dice 且带来拓扑改进

### 实验配置
- **Loss Mode**: MSE
- **Lambda 策略**: 015 (50%预训练, 30% ramp, 20%保持)
  - Phase 1 (E1-5): λ=0 (纯Dice预训练)
  - Phase 2 (E6-8): λ=0→0.1
  - Phase 3 (E9-10): λ=0.1→0.2
- **Loss Scale**: 100.0
- **ROI Mode**: fov

### 关键结果 (10 epoch)
| 指标 | Baseline | Topo | 变化 |
|------|----------|------|------|
| Val Dice | 0.4422 | 0.4711 | **+0.0289** ✅ |
| CL-Break | 55.0 | 47.6 | **-7.4 (13.5%)** ✅ |
| Max Ratio | - | 0.0745 | **< 0.1** ✅ |

### 关键发现
1. **延长纯Dice预训练至50% epochs**是关键 - 让模型先学好基础分割
2. **控制最终lambda=0.2** - 避免topo loss过强抑制Dice
3. **Ratio始终<0.1** - 符合止血阈值， topo loss不主导训练
4. **Topo Dice 超过 Baseline** - 证明topo loss带来正向收益

### 实验演进
| 配置 | Max Ratio | Val Dice | 结果 |
|------|-----------|----------|------|
| 原始(3175, loss_scale=100) | 0.08 | 0.18 | ❌ Dice过低 |
| 015策略(30%预训练) | 0.05 | 0.41 | ⚠️ 略低于Baseline |
| **015策略(50%预训练, lambda_max=0.2)** | **0.07** | **0.47** | **✅ 超过Baseline** |

### 验收状态
- ✅ Ratio < 0.1 (止血阈值)
- ✅ Dice 不降低 (实际超过Baseline)
- ✅ CL-Break 改善 13.5%
- ✅ 拓扑收益可证伪

