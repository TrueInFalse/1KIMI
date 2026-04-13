# CURRENT STATUS

> Runtime note: `train_baseline_roi.py` and `train_topo_roi.py` now support `--fast-dev` for short diagnostic runs, while default startup remains deterministic.

更新时间：2026-04-08（UTC）

## 1) 当前最强 Baseline

- 结论：`Baseline-ROI`（同口径 ROI 内 Dice）是当前最强基线。
- 证据（来自 `logs/20260405_baseline_roi_125e.csv`）：
  - best `val_dice = 0.7841`（epoch 72）
  - 125e 末尾 `val_dice = 0.7814`（epoch 125）
  - 对应拓扑指标（best 点）：`val_cl_break = 7.5`，`val_delta_beta0 = 15.0`

## 2) 当前最佳 Topo

- 结论：`Topo-Fragment-Suppress`（`fragment_suppress`）是当前最佳拓扑路线。
- 证据（来自 `logs/20260405_topo_roi_fragment_suppress_125e.csv`）：
  - best `val_dice = 0.7615`（epoch 119）
  - 125e 末尾 `val_dice = 0.7611`（epoch 125）
  - best 点拓扑指标：`cl_break = 7.8`，`delta_beta0 = 15.4`

## 3) 当前一句话判断

- 严格 125e 对照下，Topo-FS 仍落后 Baseline 约 0.02 Dice。
- 以 best 点对比：`0.7615 vs 0.7841`，Topo 仍落后 `0.0226` Dice。
- 以 final 点对比（epoch 125）：`0.7611 vs 0.7814`，Topo 仍落后 `0.0203` Dice。

## 4) 当前下一步唯一任务（指向 NEXT_ONE_THING）

- 唯一任务：在不改数据口径（Kaggle+FOV+ROI 内 Dice+125e）的前提下，只做 `fragment_suppress` 单变量收敛优化，目标把 Topo Dice 差距从 `0.0226` 缩小到 `<=0.0100`，且保持拓扑指标不退化。

## 5) 主线拓扑损失实现状态（已扶正）

- 当前默认拓扑损失实现：`topology_loss_fragment_suppress.py`
- `train_topo_roi.py` 已改为直接依赖该主线实现
- `legacy/topology_loss_ablation.py` 为历史消融保留文件，不再作为默认主线依赖

## 6) 入口一致性状态（已修正）

- `train_baseline_roi.py` 与 `train_topo_roi.py` 均支持 `--config` 参数（默认读取 `config.yaml`）。
- 两个脚本的 `--epochs` 默认值均为 `None`，仅显式传入时覆盖 YAML 的 `training.max_epochs`。
- `train_topo_roi.py` 的 `--loss-mode` 仅保留兼容入口：主线固定使用 `fragment_suppress`，非该值不会切换实际 loss。
- 规则文档：`docs/TRAIN_TOPO_ENTRY_RULES.md`。

## 7) FS 变体实验线（隔离）

- 当前主线仍然是 `fragment_suppress`，主线结论和 125e 对照结论不变。
- 新增 `thresholded_fs` 与 `topk_fs` 两条实验验证线，但它们只用于旁路验证，不得替代当前主线结论。
- 默认配置继续显式使用 `topology.variant: fragment_suppress`。
- `fragment_tau` / `fragment_topk` 只在实验配置里维护，不挂在默认主线配置里。
- 40e 快速验证配置位于 `configs/experiments/`，并通过独立输出目录和 variant-aware 文件名避免覆盖主线产物。
- 简短说明见 `reports/FS_VARIANTS_NOTE.md`。

## 30 秒口播版

- 最强 baseline 是 `Baseline-ROI`，best Dice 0.7841。
- 最佳 topo 是 `Fragment-Suppress`，best Dice 0.7615（epoch 119），125e 末尾 0.7611。
- 严格 125e 对照下，Topo-FS 相对 Baseline 仍落后约 0.02 Dice。
- 主线拓扑损失已扶正为 `topology_loss_fragment_suppress.py`；`legacy/topology_loss_ablation.py` 仅保留复盘。
