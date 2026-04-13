# Retina PH Seg

基于 U-Net 与持续同调拓扑约束的视网膜血管分割项目。

> 面向毕业设计场景：在保证像素级精度的同时，尽量改善血管连通性与碎片化问题。

---

## 项目背景

视网膜血管分割是高血压、糖尿病等慢病筛查中的关键环节。传统像素级优化（如 Dice）通常能得到可接受的重叠指标，但在细血管区域容易出现断裂、碎片化和连通性不稳定。

本项目在 U-Net 框架下引入持续同调相关拓扑正则，目标是在可复现工程流程下比较 Baseline 与 Topo 路线的真实收益。

---

## 任务定义与当前主线结论

核心任务：二分类视网膜血管分割（前景 = 血管，背景 = 非血管）。

当前默认对照口径：

- ROI 对齐（训练/验证口径一致）
- Kaggle 联合数据模式
- 统一阈值与统一日志口径

### 当前状态（主线结论）

- 当前最强结果：`Baseline-ROI 125e`（`best val_dice = 0.7841`，epoch 72；125e 末尾约 `0.7814`）
- 当前最佳 Topo：`Fragment-Suppress`（125e，`best val_dice = 0.7615`，epoch 119；125e 末尾 `val_dice = 0.7611`，epoch 125）
- 当前结论：修正 PD→loss 方向后，Topo 已具备可用性；但在严格 125e 对照下仍落后 Baseline 约 0.02 Dice

一句话：当前主线是“强 Baseline + 可用但落后约 0.02 Dice 的 Topo”。

---

## 项目整体架构与数据流

当前主线可以按“配置 -> 数据 -> 训练 -> 产物 -> 评估/可视化”理解：

```text
config.yaml / config_125e.yaml
        |
        v
data_combined.py
  |-- use_kaggle_combined = true  -> KaggleCombinedDataset
  |                                 |- 优先使用本地 data/combined
  |                                 |- 本地缺失时可走 kagglehub 自动下载
  |                                 `- 统一生成 ROI / augment / mask 匹配
  |
  `-- use_kaggle_combined = false -> data_drive.py
                                    `- 走纯 DRIVE 兼容路径
        |
        v
train_baseline_roi.py / train_topo_roi.py
  |-- model_unet.py
  |-- utils_metrics.py
  `-- topology_loss_fragment_suppress.py  (仅 Topo 主线)
        |
        v
logs/*.csv + checkpoints/*.pth
        |
        +--> evaluate.py
        |      |- 自动解析 config、数据模式、checkpoint
        |      `- 输出到 results/evaluate_<split>/
        |
        `--> visualize_results.py
               `- 训练 CSV -> results/*.png
```

从使用者视角看，最重要的是：

- 配置入口主要是 `config.yaml` 与 `config_125e.yaml`，训练/评估脚本都围绕它们读取数据模式、设备、训练轮数和输出目录。
- 数据入口主线是 `data_combined.py`；只有在 `data.use_kaggle_combined=false` 时，才退回到 `data_drive.py` 的纯 DRIVE 路线。
- Topo 训练虽然暴露了 `--loss-mode` 参数，但当前主线算法池固定为 `topology_loss_fragment_suppress.py` 中的 `fragment_suppress`。
- 训练结束后的核心产物只有三类：`checkpoints/` 下的模型权重、`logs/` 下的训练 CSV、`results/` 下的评估结果和图表。

---

## 配置与数据模式

### 配置文件

- `config.yaml`：日常主配置母版，默认入口文件。
- `config_125e.yaml`：当前 125 epoch 主线配置，其他参数沿用现有设定，`training.max_epochs=125`。

### 数据模式

项目支持两种数据模式：

#### 纯 DRIVE 模式

- 训练：21–36
- 验证：37–40
- 测试：01–20
- ROI：原始 FOV mask

#### Kaggle 联合数据模式（当前主线默认）

- 来源：DRIVE + HRF + CHASE DB1 + STARE
- 训练集：`Training/`
- 验证集：`Test/`
- 无标签测试：`Unlabeled_test/`
- ROI：由配置与加载器统一生成

与数据模式直接相关的关键开关：

- `data.use_kaggle_combined=true`：使用 `data_combined.py` 的 Kaggle 联合主线。
- `data.use_kaggle_combined=false`：切换为 `data_drive.py` 的纯 DRIVE 兼容路径。
- `data.kaggle_roi.mode`：当前主线常用 `fov`，由加载器根据图像内容估计 ROI。

---

## 指标说明

像素级指标：

- Dice
- IoU
- Precision
- Recall

结构级指标：

- CL-Break（中心线碎片数，越低越好）
- Δβ₀（连通分量差异，越低越好）

补充说明：

- `utils_metrics.py` 中的基础指标与拓扑指标都支持 ROI 掩码约束。
- 训练和评估默认采用统一阈值口径，避免“训练日志一套、评估脚本另一套”的偏差。

---

## 快速开始与常用命令

### 1) 环境准备

```bash
pip install torch torchvision
pip install segmentation-models-pytorch
pip install scikit-image scipy matplotlib tqdm pyyaml
pip install cripser
```

### 2) 选择配置

- 日常主配置：`config.yaml`
- 主线 125e：`config_125e.yaml`

### 3) 训练 Baseline-ROI

```bash
python train_baseline_roi.py --config config.yaml
```

临时覆盖训练轮数：

```bash
python train_baseline_roi.py --config config.yaml --epochs 3
```

```bash
python train_baseline_roi.py --config config.yaml --epochs 3 --fast-dev
```

- `--fast-dev` 仅用于短诊断实验；开启后会打印 `Deterministic mode: OFF` 和 `cuDNN benchmark: ON`。
- 正式主线结果建议保持默认确定性模式，不开启 `--fast-dev`。

默认行为说明：

- `--config` 默认值是 `config.yaml`。
- `--epochs` 默认值是 `None`，只有显式传入时才覆盖 YAML 中的 `training.max_epochs`。
- Baseline 训练脚本会读取 YAML 中的 `training.checkpoint_dir` 与 `training.log_dir`；默认分别是 `./checkpoints` 和 `./logs`。
- 最佳权重文件名为 `best_model_baseline_roi.pth`。

### 4) 训练 Topo-ROI

```bash
python train_topo_roi.py --config config.yaml
```

临时覆盖训练轮数：

```bash
python train_topo_roi.py --config config.yaml --epochs 3
```

```bash
python train_topo_roi.py --config config.yaml --epochs 3 --fast-dev
```

- `--fast-dev` 仅用于短诊断实验；开启后会打印 `Deterministic mode: OFF` 和 `cuDNN benchmark: ON`。
- 正式主线结果建议保持默认确定性模式，不开启 `--fast-dev`。

兼容入口示例：

```bash
python train_topo_roi.py --config config.yaml --loss-mode standard
```

默认行为说明：

- `--config` 默认值是 `config.yaml`。
- `--epochs` 默认值是 `None`，只有显式传入时才覆盖 YAML 中的 `training.max_epochs`。
- `--loss-mode` 仅保留兼容入口；当前主线固定使用 `fragment_suppress`，传入 `standard` 或 `main_component` 只会打印兼容提示，不会切换实际 loss。
- Topo 训练脚本当前将日志与权重固定输出到 `./logs` 和 `./checkpoints`。
- 产物通常包括 `best_model_topo_roi.pth` 与 `final_model_topo_roi.pth`。

详细规则可参考 `docs/TRAIN_TOPO_ENTRY_RULES.md`。

### 5) 评估

使用默认配置评估验证集：

```bash
python evaluate.py --config config.yaml --split val
```

使用 125e 配置评估验证集：

```bash
python evaluate.py --config config_125e.yaml --split val
```

显式指定权重评估：

```bash
python evaluate.py --config config.yaml --checkpoint checkpoints/best_model_baseline_roi.pth --split val
```

默认行为说明：

- `--config` 默认值是 `config.yaml`。
- `--split` 支持 `train`、`val`、`test`，默认值是 `val`。
- `--checkpoint` 可选；若不传，脚本会根据 `config` 中的数据模式和 Topo 开关，自动去 `training.checkpoint_dir` 搜索候选权重。
- Kaggle 主线下，若 Topo 开启，会优先尝试 `best_model_topo_roi.pth`，其次尝试 `best_model_baseline_roi.pth`，再回退到其他候选文件。
- 评估输出会写入 `results/evaluate_<split>/`，其中包含预测图目录 `predictions/` 和总览图 `evaluation_results.png`。
- 对无标签 split，脚本会跳过指标计算，但仍可产出可视化结果。

### 6) 训练日志可视化

输入：任意训练 CSV，路径由 `--log-file` 指定，不写死 `logs/`。

输出：默认 `results/<csv_stem>.png`，可通过 `--output` 自定义。

```bash
python visualize_results.py --log-file logs/20260405_baseline_roi_125e.csv --mode auto
python visualize_results.py --log-file logs/20260405_topo_roi_fragment_suppress_125e.csv --title "Topo ROI Fragment-Suppress"
```

常用参数：

- `--log-file`：必填，训练日志 CSV 路径。
- `--mode`：`auto | baseline | topo`，默认 `auto`，会根据列名自动推断日志类型。
- `--output`：输出 PNG 路径，默认 `results/<csv_stem>.png`。
- `--title`：可选，自定义图标题。
- `--no-overwrite`：若输出文件已存在则直接报错，避免覆盖旧图。

### 7) 常用入参速查

| 脚本 | 参数 | 默认值 | 实际作用 |
| --- | --- | --- | --- |
| `train_baseline_roi.py` | `--config` | `config.yaml` | 读取训练、数据、设备、输出目录配置 |
| `train_baseline_roi.py` | `--epochs` | `None` | 仅在显式传入时覆盖 `training.max_epochs` |
| `train_topo_roi.py` | `--config` | `config.yaml` | 读取训练、数据与拓扑相关配置 |
| `train_topo_roi.py` | `--epochs` | `None` | 仅在显式传入时覆盖 `training.max_epochs` |
| `train_topo_roi.py` | `--loss-mode` | `fragment_suppress` | 兼容参数，主线不会切换实际 topo loss |
| `evaluate.py` | `--config` | `config.yaml` | 决定数据模式、设备与 checkpoint 搜索目录 |
| `evaluate.py` | `--checkpoint` | `None` | 显式指定权重；为空时自动搜寻最佳模型 |
| `evaluate.py` | `--split` | `val` | 评估 `train / val / test` 之一 |
| `visualize_results.py` | `--log-file` | 无 | 指定输入 CSV |
| `visualize_results.py` | `--mode` | `auto` | 自动或手动指定 baseline/topo 日志模式 |
| `visualize_results.py` | `--output` | `results/<csv_stem>.png` | 指定输出 PNG |
| `visualize_results.py` | `--no-overwrite` | `false` | 防止覆盖已存在图像 |

---

## 核心脚本功能词典

下表聚焦根目录主线 `.py` 文件，按“谁负责什么”快速建立认知：

| 文件 | 角色 | 说明 |
| --- | --- | --- |
| `train_baseline_roi.py` | Baseline 训练入口 | 当前 Baseline 主训练脚本，负责读取配置、构建模型、执行 ROI 内 Dice loss 训练，并保存 `best_model_baseline_roi.pth`。训练与验证都遵循 ROI 对齐口径。 |
| `train_topo_roi.py` | Topo 训练入口 | 当前 Topo 主训练脚本，串联 `model_unet.py`、`data_combined.py`、`utils_metrics.py` 与 `topology_loss_fragment_suppress.py`。虽然提供 `--loss-mode` 参数，但主线实际固定使用 `fragment_suppress`。 |
| `evaluate.py` | 统一评估入口 | 用于训练后评估与可视化，支持 `--config`、`--checkpoint`、`--split`。未显式传入 `--checkpoint` 时，会自动根据配置和目录内容推断并加载最合适的最佳模型。 |
| `visualize_results.py` | 日志图表生成工具 | 将训练 CSV 日志绘制为 2x3 曲线图，支持自动识别 Baseline/Topo 日志模式，并允许自定义标题、输出路径与覆盖策略。 |
| `data_combined.py` | 数据主线入口 | 当前 Kaggle/联合数据主线归它负责；它根据 `use_kaggle_combined` 选择 Kaggle 联合加载或回退 DRIVE。Kaggle 路线同时承担本地数据优先、自动下载、ROI 生成、mask 匹配与增强逻辑。 |
| `data_drive.py` | DRIVE 兼容数据入口 | 这是纯 DRIVE 路线的“归口文件”，处理 16+4 划分、原始 FOV ROI 与标签读取。它不是当前默认主线，但仍是 `data_combined.py` 在 DRIVE 模式下调用的权威实现。 |
| `utils_metrics.py` | 指标工具库 | 统一提供 Dice、IoU、Precision、Recall、CL-Break、Δβ₀ 等指标计算，并明确支持 ROI 掩码约束。训练脚本与 `evaluate.py` 都依赖它保持指标口径一致。 |
| `topology_loss_fragment_suppress.py` | Topo 算法池 | 当前主线拓扑正则核心实现，封装 `Fragment-Suppress` 逻辑、ROI 约束与持续同调计算。只要你在跑当前 Topo 主线，真正生效的 topo loss 就在这里。 |
| `model_unet.py` | 网络构建工具 | 负责创建基于 `segmentation-models-pytorch` 的 U-Net，并支持加载本地 ResNet34 预训练权重。评估阶段的 `load_model()` 也由它负责根据 checkpoint 恢复模型。 |

---

## 仓库结构（按职责而非时间线）

```text
.
├── README.md
├── config.yaml
├── config_125e.yaml
├── data_combined.py
├── data_drive.py
├── model_unet.py
├── train_baseline_roi.py
├── train_topo_roi.py
├── topology_loss_fragment_suppress.py
├── utils_metrics.py
├── evaluate.py
├── visualize_results.py
├── checkpoints/
├── logs/
├── results/
├── docs/
├── reports/
├── configs/archive/
├── legacy/
└── pretrained_weights/
```

建议按下面的职责去理解目录，而不是在仓库里“跳着找”：

- 根目录主线入口：训练、评估、数据加载、模型定义和指标工具都在根目录。
- `checkpoints/`、`logs/`、`results/`：运行产物目录，分别保存权重、训练日志和评估/可视化结果。
- `docs/`：方法说明、仓库整理记录、当前状态、环境模板与训练入口约束等辅助文档。
- `reports/`：阶段性实验结论与汇报材料。
- `configs/archive/`：历史阶段配置，不作为当前主线默认入口。
- `legacy/`：历史脚本与消融保留文件，仅用于复盘，不作为默认主线路径。

---

## 主线与历史文件的边界

默认推荐只关注以下主线文件：

- `config.yaml` / `config_125e.yaml`
- `data_combined.py`
- `train_baseline_roi.py`
- `train_topo_roi.py`
- `topology_loss_fragment_suppress.py`
- `evaluate.py`
- `visualize_results.py`

## Fragment-Suppress Variants (Experimental Only)

- Reference mainline stays `fragment_suppress`. Its math definition and default behavior remain unchanged.
- `thresholded_fs` and `topk_fs` are isolated validation branches only. They do not replace current README conclusions or 125e mainline results.
- Default `config.yaml` and `config_125e.yaml` now explicitly pin `topology.variant: fragment_suppress`.
- `fragment_tau` and `fragment_topk` are maintained in experiment configs only, not in default mainline configs.
- Variant-specific quick-check configs live under `configs/experiments/`:
  - `config_thresholded_fs_tau001_40e.yaml`
  - `config_thresholded_fs_tau002_40e.yaml`
  - `config_topk_fs_top4_40e.yaml`
  - `config_topk_fs_top8_40e.yaml`
- Experimental outputs are isolated with variant-aware filenames and experiment-specific log/checkpoint directories.
- Short note: `reports/FS_VARIANTS_NOTE.md`

## Runtime Notes

- `train_baseline_roi.py` and `train_topo_roi.py` now support `--fast-dev` for short diagnostic runs.
- Default startup behavior remains deterministic: `Deterministic mode: ON` and `cuDNN benchmark: OFF`.
- With `--fast-dev`, startup switches to `Deterministic mode: OFF` and `cuDNN benchmark: ON`.
- Keep the default deterministic mode for formal mainline results and paper-facing comparisons.

历史与归档文件说明：

- `legacy/topology_loss_ablation.py` 为历史消融保留文件，不作为默认主线依赖。
- 其他历史入口与旧配置已归档到 `legacy/` 或 `configs/archive/`。
- 如果你的目标是“复现实验主线”，优先使用根目录主线文件，不必从 `legacy/` 开始读起。
