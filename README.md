# Retina PH Seg - 视网膜血管分割项目

基于持续同调（Persistent Homology）的U-Net视网膜血管分割系统，解决血管断裂问题。

## 项目概述

**研究目标**：通过拓扑约束（PH Loss）改善U-Net在视网膜血管分割中的连通性问题。

**当前阶段**：Stage 2 - 端到端拓扑正则训练（cripser可微持续同调）

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

### 6. λ调度策略（可扩展）

`train_with_topology.py` 支持通过 `config.yaml -> topology.lambda_schedule` 选择调度策略：

- `015`（默认）：前30% epochs 固定λ=0；中间30%线性0→0.1；最后40%线性0.1→0.5。
- `3175`：前30轮固定λ=0.1；后70轮线性增至0.5；剩余轮次固定0.5。

后续新增策略时，只需在代码中新增策略分支并在配置中追加参数，不必覆盖旧策略。

## 性能基准

| 指标 | Stage 1基线 | Stage 2目标（+PH Loss） |
|------|-------------|------------------------|
| Val Dice | ~0.75-0.78 | ≥0.82 |
| CL-Break | ~70 | <10 |
| Δβ₀ | ~200 | ~0 |

## 文档说明

- 历史修复与变更记录已迁移到 `FIX_LOG.md`，README仅保留使用说明与架构概览。
