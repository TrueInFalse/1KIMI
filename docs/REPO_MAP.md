# REPO MAP

> Runtime note: the mainline training entries `train_baseline_roi.py` and `train_topo_roi.py` keep deterministic defaults and additionally expose `--fast-dev` for short diagnostic runs.

更新时间：2026-04-08（UTC）

## 范围说明

- 本地图覆盖仓库中的项目文件，不含 `.git/` 内部对象。
- 数据集大文件（如 `DRIVE/**`、`data/combined/**`）用目录级条目归类，不逐张图片展开。
- 本文只做仓库地图与轻量整理记录，不涉及训练逻辑改动。

## 1) `current`（当前主线正在使用）

| 文件/路径 | 一句话用途 |
|---|---|
| `README.md` | 当前主线总入口（状态、主线、快速开始）。 |
| `docs/REPO_MAP.md` | 仓库地图与分层管理说明。 |
| `docs/MOVE_LOG.md` | 文件迁移记录与路径风险记录。 |
| `docs/CURRENT_STATUS.md` | 当前最强 baseline / 最佳 topo / 下一步任务快照。 |
| `docs/NEXT_ONE_THING.md` | 当前唯一任务与完成判据。 |
| `docs/MAINLINE_FILES.md` | 主线文件快速导航。 |
| `docs/ENVIRONMENT_TEMPLATE.md` | 环境快照模板（已补全当前容器信息）。 |
| `docs/TRAIN_TOPO_ENTRY_RULES.md` | Topo 主线入口规则（config/epochs/loss_mode 优先级）。 |
| `docs/0307structure.txt` | 项目结构参考文本。 |
| `config.yaml` | 主配置母版（带注释）。 |
| `config_125e.yaml` | 主线 125e 配置。 |
| `data_combined.py` | 主线数据加载器（Kaggle 联合 + DRIVE 双模式）。 |
| `model_unet.py` | U-Net 模型定义。 |
| `utils_metrics.py` | 指标计算（像素+拓扑）。 |
| `topology_loss_fragment_suppress.py` | 当前默认主线拓扑损失实现。 |
| `train_baseline_roi.py` | 主线 Baseline-ROI 训练入口（支持 `--config`/`--epochs`）。 |
| `train_topo_roi.py` | 主线 Topo-ROI 训练入口（支持 `--config`/`--epochs`；主线固定 fragment_suppress）。 |
| `evaluate.py` | 评估脚本入口。 |
| `visualize_results.py` | 主线训练日志曲线可视化入口（2x3 子图）。 |
| `logs/20260405_baseline_roi_125e.csv` | 当前基线主证据日志（125e）。 |
| `logs/20260405_topo_roi_fragment_suppress_125e.csv` | 当前最佳 topo 主证据日志（125e）。 |
| `reports/MAINLINE_RESULTS_SUMMARY.md` | 当前主线统一摘要（论文/README/PPT 共用口径）。 |
| `reports/BETTI0_AUDIT_REPORT.md` | Betti0 per-image audit summary using independent samples deduplicated by `base_image_id`. |
| `checkpoints/best_model_baseline_roi.pth` | 当前最强 Baseline 权重。 |
| `checkpoints/best_model_topo_roi.pth` | Topo-ROI 最佳权重。 |
| `checkpoints/final_model_topo_roi.pth` | Topo-ROI 最终权重。 |
| `reports/FRAGMENT_SUPPRESS_125E_REPORT.md` | 主线 topo 结论报告。 |
| `reports/ROI_ALIGNED_20E_REPORT.md` | ROI 对齐阶段报告。 |
| `reports/TOPO_SHAPE_ABLATION_20E_REPORT.md` | Topo 消融阶段报告。 |
| `reports/ROI_AUDIT_REPORT.md` | ROI 审计报告。 |
| `reports/TOPO_ROI_EVIDENCE.md` | Topo ROI 证据报告。 |
| `pretrained_weights/resnet34-333f7ec4.pth` | 编码器预训练权重。 |
| `DRIVE/**` | DRIVE 数据集原始图像与标注。 |
| `data/combined/**` | Kaggle 联合数据目录。 |

## 2) `archive`（历史实验/旧版本/保留记录）

| 文件/路径 | 一句话用途 |
|---|---|
| `configs/archive/config_20e.yaml` | 20e 历史阶段配置。 |
| `legacy/data_drive.py` | 旧数据加载脚本归档。 |
| `legacy/topology_loss.py` | 旧拓扑损失模块归档。 |
| `legacy/train_baseline.py` | 旧 Baseline 训练入口归档。 |
| `legacy/train_with_topology.py` | 旧 Topology 训练入口归档。 |
| `legacy/项目结构260315.png` | 历史结构图归档。 |
| `legacy/topology_loss_ablation.py` | 历史消融实现保留文件（非默认主线依赖）。 |
| `experiments/roi_aligned_20e/*` | ROI 20e 阶段实验快照（脚本+日志）。 |
| `experiments/topo_shape_ablation/*` | Topo 消融阶段材料与日志。 |
| `logs/training_topo_roi_20e_complete.csv` | 20e Topo-ROI 阶段日志。 |
| `logs/training_topo_roi_log.csv` | ROI Topo 早期日志。 |
| `logs/training_topo_log.csv` | 旧 topo 路线日志。 |
| `logs/training_log.csv` | 旧 baseline 日志。 |
| `checkpoints/best_model.pth` | 早期 baseline 权重。 |
| `checkpoints/best_model_topo.pth` | 早期 topo 权重。 |
| `checkpoints/final_model_topo.pth` | 早期 topo 最终权重。 |
| `checkpoints/checkpoint_epoch_*.pth` | 历史中间 epoch 权重归档。 |
| `checkpoints/stage1_baseline_frozen.pth` | Stage1 冻结权重归档。 |
| `results/**` | 历史可视化与推理产物。 |
| `audit_results/**` | 历史 ROI 审计产物。 |
| `回收站/**` | 旧文档与历史资料归档区。 |
| `.ipynb_checkpoints/*.py` | Notebook 自动保存脚本。 |
| `__pycache__/**` | Python 缓存文件。 |

## 3) `utility`（审计脚本/一次性工具/辅助文件）

| 文件/路径 | 一句话用途 |
|---|---|
| `test/roi_audit_final.py` | ROI 链路最终审计脚本。 |
| `test/roi_audit_kaggle.py` | Kaggle 口径 ROI 审计脚本。 |
| `test/audit_roi_dice_gap.py` | ROI Dice 差异审计脚本。 |
| `test/verify_data.py` | 数据加载与样本校验脚本。 |
| `test/verify_roi_usage.py` | ROI 使用一致性检查脚本。 |
| `test/threshold_sweep.py` | 阈值扫描分析脚本。 |
| `test/test_topo_fix.py` | Topo 模块修复验证脚本。 |
| `test/test_end_to_end.py` | 端到端导入/路径检查脚本。 |
| `experiments/audit_betti0_per_image.py` | Betti0 audit utility that removes duplicate samples by `base_image_id` before writing CSV/report/visualizations. |
| `test/run_topo_roi_debug.sh` | Topo ROI 调试脚本。 |
| `test/verify_user.py` | 用户验证辅助脚本。 |
| `test/verify_user.ipynb` | 交互式验证 notebook。 |
| `test/test_0211_01_u.py` | 历史测试脚本留存。 |
| `artifacts/roi_audit/*` | 审计中间证据与可视化。 |
| `logs/comparison_report.txt` | 对比汇总文本输出。 |
| `logs/pd_debug_20260305.txt` | 持续同调调试日志。 |
| `logs/tta_summary.json` | TTA 摘要统计。 |
| `data/archive.zip` | 数据打包归档文件。 |
| `.vscode/settings.json` | 本地 IDE 辅助配置。 |
| `.gitignore` | Git 忽略规则。 |

## 4) 配置文件用途（当前状态）

| 配置文件 | 当前用途 | 状态 |
|---|---|---|
| `config.yaml` | 主配置母版（默认入口）。 | current |
| `config_125e.yaml` | 主线 125e 配置。 | current |
| `configs/archive/config_20e.yaml` | 20e 历史对照配置。 | archive |

## 4.5) FS 变体实验线（隔离）

- `topology_loss_fragment_suppress.py`
  - keeps `fragment_suppress` as the frozen reference mainline
  - adds opt-in experimental branches `thresholded_fs` and `topk_fs`
- `configs/experiments/`
  - `config_thresholded_fs_tau001_40e.yaml`
  - `config_thresholded_fs_tau002_40e.yaml`
  - `config_topk_fs_top4_40e.yaml`
  - `config_topk_fs_top8_40e.yaml`
  - `fragment_tau` / `fragment_topk` are maintained here, not in default `config.yaml` or `config_125e.yaml`
- `reports/FS_VARIANTS_NOTE.md`
  - short note clarifying these branches are validation-only and do not overwrite current mainline claims
- Output convention for `train_topo_roi.py`
  - variant-aware CSV/checkpoint filenames
  - `fragment_suppress` also keeps legacy aliases for existing mainline tooling

## 5) 轻量整理执行结果（截至当前）

- 第一阶段：文档/报告/20e 配置已收束到 `docs/`、`reports/`、`configs/archive/`。
- 追加阶段：旧主入口脚本收束到 `legacy/`（不改逻辑、不改 import、不删文件）。
- 主线 topo loss 扶正：默认实现已切换为 `topology_loss_fragment_suppress.py`。
- `legacy/topology_loss_ablation.py` 作为历史消融文件保留，不再作为默认主线依赖。
- 日志可视化已扶正：`visualize_results.py` 迁回根目录并作为主线入口。
- 详细逐条迁移见 `docs/MOVE_LOG.md`。
