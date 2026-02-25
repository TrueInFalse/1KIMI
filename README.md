# Retina PH Seg - 视网膜血管分割项目

基于持续同调（Persistent Homology）的U-Net视网膜血管分割系统，解决血管断裂问题。

## 项目概述

**研究目标**：通过拓扑约束（PH Loss）改善U-Net在视网膜血管分割中的连通性问题。

**当前阶段**：Stage 2 - 端到端拓扑正则训练（STE + CubicalRipser）

**数据集**：

项目支持**双模式**数据集（通过`config.yaml`中`data.use_kaggle_combined`切换）：

#### 模式A：纯DRIVE（默认）
- **数据集**: DRIVE（Digital Retinal Images for Vessel Extraction）
- **训练集**: 16张（ID: 21-36）
- **验证集**: 4张（ID: 37-40）
- **测试集**: 20张（ID: 01-20，无标签）
- **ROI**: 使用原始FOV掩码（圆形视野）

#### 模式B：Kaggle联合数据集
- **数据集**: [Retinal Vessel Segmentation Combined](https://www.kaggle.com/datasets/pradosh123/retinal-vessel-segmentation-combined)
- **组成**: DRIVE + HRF + CHASE DB1 + STARE
- **训练集**: `train/` 目录（多数据集混合，约60+张）
- **验证集**: `test/` 目录（已有官方划分，约20+张）
- **测试集**: `unlabeled_test/` 目录（仅图像，无GT）
- **ROI**: 自动生成全1 mask（混合数据集无统一FOV）
- **自动下载**: 首次使用自动调用`kagglehub`下载

**切换方法**:
```yaml
# config.yaml
data:
  use_kaggle_combined: false   # false = 纯DRIVE
  # use_kaggle_combined: true  # true = Kaggle联合数据集
```

**依赖安装**（联合数据集模式需要）:
```bash
pip install kagglehub
```

## 文件结构

### 核心Python模块（7个）

| 文件 | 功能 | 说明 |
|------|------|------|
| `config.yaml` | 项目配置 | 数据路径、训练参数、模型参数、数据集模式切换 |
| `data_drive.py` | 数据加载器 | DRIVE数据集加载，16+4划分，ROI约束 |
| `data_combined.py` | 联合数据加载器 | 双模式支持：纯DRIVE / Kaggle联合数据集（DRIVE+CHASE+STARE+HRF） |
| `model_unet.py` | U-Net模型 | smp.Unet + ResNet34编码器，支持本地权重 |
| `utils_metrics.py` | 评估指标 | Dice/IoU/Precision/Recall + CL-Break/Δβ₀拓扑指标 |
| `train_baseline.py` | 训练脚本 | 纯Dice Loss，早停，学习率调度 |
| `evaluate.py` | 评估脚本 | 验证/测试集评估，可视化结果 |
| `visualize_results.py` | 可视化 | 验证集对比图、测试集预测图、训练曲线 |
| `topology_loss.py` | 拓扑损失 | 软Betti数损失（端到端训练使用） |

### 配套文档（MD文档/目录）

每个核心模块都有对应的`.md`文档，说明接口定义、使用示例和依赖关系。

### 输出目录

```
./
├── checkpoints/
│   └── best_model.pth          # 最佳模型权重
├── logs/
│   ├── training_log.csv        # 训练日志
│   └── training_curves.png     # 训练曲线图
└── results/
    ├── val_sample_comparison.png   # 验证集对比图
    ├── test_sample_prediction.png  # 测试集预测图
    └── predictions/            # 预测概率图（.npy）
```

## 快速开始

### 环境准备


**依赖安装**：
```bash
pip install torch torchvision
pip install segmentation-models-pytorch
pip install scikit-image scipy
pip install pyyaml tqdm matplotlib
```

**版本控制**：

- **SSH 密钥认证 + 直连 GitHub**（推荐）：已完成配置

- Clash代理不再启用：
    Clash 崩溃后快速恢复（30秒修复）。如果推送时卡住/超时，按此顺序执行：

    ```bash
    # Step 1: 确认死亡
    curl --socks5 127.0.0.1:7890 -I https://github.com
    # 如果无输出或 "Connection refused"，进入 Step 2

    # Step 2: 清理僵尸进程（如果有）
    pkill -f clash-meta 2>/dev/null

    # Step 3: 重启
    cd ~/clash
    nohup ./clash-meta -f ~/.config/clash/config.yaml > clash.log 2>&1 &

    # Step 4: 验证（等待2秒）
    sleep 2 && curl --socks5 127.0.0.1:7890 -I https://github.com 2>/dev/null | head -1

    # Step 5: 重新推送
    cd ~/autodl-tmp/1KIMI && git push
    ```

### 数据准备

#### 纯DRIVE模式（默认）

将DRIVE数据集放在项目根目录：
```
DRIVE/
├── training/
│   ├── images/           # 训练图像 *.tif
│   ├── 1st_manual/       # 血管标签 *.gif
│   └── mask/             # ROI掩码 *.gif
└── test/
    ├── images/           # 测试图像 *.tif
    └── mask/             # ROI掩码 *.gif
```

#### Kaggle联合数据集模式

**数据加载优先级**（代码自动处理）：
1. **优先检查本地**: `data/combined/Training/images/` 是否存在且非空
2. **如不存在**: 自动调用 `kagglehub.dataset_download()` 下载

**本地数据结构**（如已手动准备）：
```
data/combined/
├── Training/           # 训练集
│   ├── images/         # 训练图像
│   └── masks/          # 血管标签
├── Test/               # 验证集
│   ├── images/         # 验证图像
│   └── masks/          # 验证标签
└── Unlabeled_test/     # 无标签测试集（可选）
```

**自动下载方式**（本地不存在时）：
```bash
# 1. 安装kagglehub
pip install kagglehub

# 2. 修改配置
# config.yaml -> data.use_kaggle_combined: true

# 3. 运行训练（首次自动下载到 ~/.cache/kagglehub/）
python train_baseline.py  # 或 train_with_topology.py
```

### 预训练权重

ResNet34 ImageNet预训练权重已下载到本地：
```
pretrained_weights/
└── resnet34-333f7ec4.pth   # 本地预训练权重
```

首次运行会自动加载本地权重，无需网络请求。

### 训练模型

```bash
python train_baseline.py
```

训练结束后会生成：
- `checkpoints/best_model.pth`：最佳模型
- `logs/training_log.csv`：训练日志
- `logs/training_curves.png`：训练曲线

### 评估模型

```bash
# 验证集评估
python evaluate.py --split val

# 测试集评估（无标签，仅预测）
python evaluate.py --split test
```

### 可视化结果

```bash
# 生成验证集对比图和测试集预测图
python visualize_results.py
```

## 关键特性

### 1. 数据验证

- **16+4划分**：严格从training文件夹划分16张训练+4张验证
- **路径检查**：自动区分`1st_manual`（血管）和`mask`（ROI）
- **数值验证**：血管标签均值~10%，ROI均值~70%

### 2. ROI约束

所有指标（Dice、IoU、CL-Break等）均在ROI区域内计算：
```python
pred_roi = pred[roi_mask > 0]
target_roi = target[roi_mask > 0]
```

### 3. 拓扑指标

- **CL-Break**：中心线碎片数（越低越好，基线~70，目标<10）
- **Δβ₀**：Betti数误差（连通分量数差异，目标~0）

### 4. 数据增强

训练时自动应用以下数据增强（同步应用于图像和标签）：
- 随机水平翻转（50%概率）
- 随机垂直翻转（50%概率）
- 随机90度旋转（30%概率，90/180/270度）
- 随机亮度调整（30%概率，0.8-1.2倍）
- 随机对比度调整（20%概率，0.8-1.2倍）

### 5. 训练监控

- 每轮显示所有指标（Loss、Dice、IoU、Prec、Rec、CL-Break、Δβ₀）
- 显示每轮用时、总用时、预计剩余时间
- 早停基于Val Dice（patience=20）
- 自动生成训练曲线图（logs/training_curves.png）

## 性能基准

| 指标 | Stage 1基线 | Stage 2目标（+PH Loss） |
|------|-------------|------------------------|
| Val Dice | ~0.75-0.78 | ≥0.82 |
| CL-Break | ~70 | <10 |
| Δβ₀ | ~200 | ~0 |

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

