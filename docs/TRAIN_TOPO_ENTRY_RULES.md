# TRAIN TOPO ENTRY RULES

更新时间：2026-04-08（UTC）

## 1) 入口一致性（Baseline / Topo）

- `train_baseline_roi.py` 与 `train_topo_roi.py` 都支持 `--config`，默认值都是 `config.yaml`。
- `--epochs` 在两个脚本中默认都是 `None`，仅显式传入时覆盖 YAML 的 `training.max_epochs`。
- 因此：不传 `--epochs` 时按 YAML 生效；传入 `--epochs` 时以 CLI 为准。

## 2) 默认拓扑损失实现

- 当前默认主线实现：`topology_loss_fragment_suppress.py`
- 历史消融保留文件：`legacy/topology_loss_ablation.py`（不作为默认主线依赖）

## 3) Topo 脚本的 `--loss-mode` 真实行为

- 参数：`--loss-mode`
- 默认值：`fragment_suppress`
- 兼容可选值：`standard`、`main_component`、`fragment_suppress`
- 主线行为：训练时固定使用 `fragment_suppress`。
- 说明：传入 `standard`/`main_component` 仅输出兼容提示，不会切换实际 loss 数学逻辑。

## 3.5) `topology.variant` 当前行为

- 实际拓扑分支由 config 中的 `topology.variant` 决定。
- 默认主线值仍然是 `fragment_suppress`。
- 可选实验值：
  - `thresholded_fs`
  - `topk_fs`
- `fragment_tau` 仅在 `thresholded_fs` 下生效。
- `fragment_topk` 仅在 `topk_fs` 下生效。
- `--loss-mode` 继续只作为旧入口兼容参数，不负责切换当前实验分支。

## 4) 当前推荐运行命令（主线）

```bash
python train_baseline_roi.py --config config.yaml
python train_topo_roi.py --config config.yaml
```

可选（临时覆盖轮数）：

```bash
python train_baseline_roi.py --config config.yaml --epochs 3
python train_topo_roi.py --config config.yaml --epochs 3
```

可选（兼容提示入口，不会切换主线 loss）：

```bash
python train_topo_roi.py --config config.yaml --loss-mode standard
```
