# 全局代码实现审计报告（正确性导向）

审计日期：2026-03-15  
审计范围：训练/验证/评估主链路、dataset/ROI、topology loss、指标实现、实验可比性

## A. 项目真实主线梳理

### 主训练入口（当前实际主线）
1. **Baseline-ROI**：`train_baseline_roi.py`，使用 `get_combined_loaders(config)`，训练损失为 ROI 内 Dice。  
2. **Topology-ROI**：`train_topo_roi.py`，同样使用 `get_combined_loaders(config)`，总损失 `L = Dice_ROI + λ * Topo`。  
3. 旧分支仍可运行：`train_baseline.py`（全图 Dice）与 `train_with_topology.py`（全图 Dice + Topo）。

### 主验证/评估入口
1. 训练内验证：四个 train 脚本各自的 `validate()`。  
2. 独立评估脚本：`evaluate.py`，但**仅走 DRIVE loader**，与 Kaggle 联合主线不一致。

### 关键依赖文件
- 数据：`data_combined.py`（Kaggle/DRIVE双模式入口），`data_drive.py`。  
- 模型：`model_unet.py`。  
- 拓扑损失：`topology_loss.py`、`topology_loss_ablation.py`。  
- 指标：`utils_metrics.py`。

### 当前主结论实际依赖链路（按代码执行）
`config.yaml(use_kaggle_combined=true)` → `data_combined.get_combined_loaders` → `train_baseline_roi.py / train_topo_roi.py` → `utils_metrics.py` 产生日志 CSV（`logs/*.csv`）→ `experiments/roi_aligned_20e/*.md` 结论。

---

## B. 已确认的问题（高确定性）

### 1) Topo-ROI 默认强制 20 epoch，导致与配置口径不一致
- **位置**：`train_topo_roi.py::main()`
- **事实**：`--epochs` 默认值写死为 20，且 `if args.epochs:` 恒成立，覆盖 `config['training']['max_epochs']`。  
- **为什么是问题**：用户以为按 `config.yaml`（如125）训练，实际常被固定为20。  
- **后果**：实验对比不公平、结论不可比（尤其和 Baseline 跑满配置时）。
- **严重程度**：**高**。

### 2) Kaggle 模式把 `Test/` 当验证集用于调参，存在“测试集污染”
- **位置**：`data_combined.py::_get_kaggle_combined_loaders()`
- **事实**：训练集取 `Training/`，验证集直接取 `Test/`。  
- **为什么是问题**：若该 Test 语义是公开测试集，频繁用于选择模型/阈值即发生评估污染。  
- **后果**：报告指标偏乐观，泛化结论不可信。  
- **严重程度**：**高**。

### 3) `evaluate.py` 与当前 Kaggle 主线评估链路脱节
- **位置**：`evaluate.py::main()`
- **事实**：评估固定调用 `get_drive_loaders(config_path)`，不走 `get_combined_loaders(config)`。  
- **为什么是问题**：训练用 Kaggle，离线评估却默认 DRIVE，口径不一致。  
- **后果**：容易出现“训练看起来在A，评估实际在B”。  
- **严重程度**：**高**。

### 4) DRIVE test 集无 GT，却仍可被评估脚本计算 Dice 等指标
- **位置**：`data_drive.py::__getitem__` 与 `evaluate.py`。
- **事实**：test 模式返回全零 `vessel_mask` 占位；`evaluate.py --split test` 仍计算 Dice/IoU。  
- **为什么是问题**：指标没有语义有效性。  
- **后果**：可能输出“看似正常的数值”但实为无效评估。  
- **严重程度**：**高**。

### 5) 指标异常时返回 -1，被上游直接平均，可能污染结论
- **位置**：`utils_metrics.py::compute_topology_metrics()` 及各训练脚本 `validate()`。
- **事实**：拓扑计算异常时返回 `cl_break=-1, delta_beta0=-1`；上层多数代码直接累加平均。  
- **为什么是问题**：错误值混入均值会系统性偏移结论。  
- **后果**：拓扑指标可能被“人为拉好”或异常波动。  
- **严重程度**：**中-高**。

### 6) `data_combined.get_combined_loaders` 在 DRIVE 分支忽略传入 config
- **位置**：`data_combined.py::get_combined_loaders()`
- **事实**：当 `use_kaggle_combined=false` 时，调用 `_get_drive_loaders_original(config_path='config.yaml')`，不是使用传入的 `config`。  
- **为什么是问题**：外部脚本运行时覆盖参数（batch size/img_size/ids）可能失效。  
- **后果**：复现实验时出现“改了配置但不生效”。  
- **严重程度**：**中**。

### 7) README/报告宣称的 clDice 在代码主链路中不存在
- **位置**：全仓库 Python 主线（train/evaluate/utils）
- **事实**：无 clDice 实现/调用。  
- **为什么是问题**：若结论口径包含 clDice，则实现证据缺失。  
- **后果**：论文或汇报中该指标结论不可追溯。  
- **严重程度**：**中**。

---

## C. 高风险疑点（暂未完全证实）

### 1) Topology loss 可能主要在优化“有限寿命碎片”，而非主连通结构
- **可疑点**：`topology_loss*.py` 只使用 finite deaths；H0 的全局主分量常是 infinite death，可能被排除。  
- **为何可疑**：若主分量不进入损失，优化可能偏向次级碎片寿命分布。  
- **缺失证据**：需对单样本 PD（含 infinite 条）做逐轮跟踪并与梯度方向对齐。  
- **最小验证**：在单 batch 打印 `num_finite`、最大 finite lifetime 与分割连通性变化关系。

### 2) `target_beta0=5` + top-k lifetime 目标可能与血管树拓扑不匹配
- **可疑点**：标准损失把前 k 个 lifetime 拉向固定 target。  
- **为何可疑**：可能鼓励多个“稳定组件”并存，不一定鼓励单一连通主干。  
- **缺失证据**：需做 `target_beta0`/`target_lifetime` 消融并检查 CL-Break、β0 曲线。  
- **最小验证**：固定数据划分，跑 5/10 epoch 快速对比。

### 3) `_find_mask` 的模糊匹配存在错配风险
- **可疑点**：`img_stem in mask_stem or mask_stem in img_stem` 的回退匹配可能一对多命中。  
- **为何可疑**：一旦命名不规范，可能图像与标签错配且无报错。  
- **缺失证据**：需要真实数据目录进行“image→mask唯一映射”统计。  
- **最小验证**：输出每个样本映射，检查重复 mask/未命中。

### 4) Topo 指标异常处理在不同脚本不一致
- **可疑点**：`train_topo_roi` 出错时填 0；`utils_metrics` 出错时返回 -1；Baseline 直接使用返回值。  
- **为何可疑**：同一指标跨实验脚本分布可被错误处理策略改变。  
- **缺失证据**：需要触发一次异常并比较各脚本日志差异。  
- **最小验证**：构造非法输入/空图测试 `compute_topology_metrics` 并对比日志。

---

## D. 审计后认为基本合理的部分

1. **训练循环顺序基本正确**：`model.train/eval`、`zero_grad → forward → backward → step`、scheduler 每 epoch 更新，整体无明显反序错误。  
2. **ROI 对齐版损失口径改动是实质接入**：`compute_dice_loss_roi` 被真实用于反向传播，而非仅日志。  
3. **随机种子设置较完整**：`random/numpy/torch/cudnn` 均设置，具备基础复现能力。  
4. **模型输出与 Dice from_logits 口径匹配**：主干输出 logits，评估侧显式 sigmoid + threshold。

---

## E. 对实验结论可信度判断（代码实现视角）

### 目前相对可信
- “当前 Topology-ROI 在既有参数下明显不优于 Baseline-ROI”这一趋势，**在代码层面是可能成立的**（因为 topo loss 确实参与了优化）。

### 需要重算/复核
1. 任何使用 Kaggle `Test/` 持续调参得到的最优结果（存在评估污染风险）。  
2. 跨脚本比较（`train_*_roi` vs `train_with_topology` vs `evaluate.py`）的结论（评估链路不统一）。  
3. 包含拓扑指标均值的结论（异常值处理不一致/可能掺入 -1 或 0）。

### 暂时不能直接相信
- 涉及 clDice 的结论（主线代码无实现证据）。  
- 将 `evaluate.py` 的输出直接当作 Kaggle 主线最终报告的结论。

---

## F. 最高优先级修复清单（按优先级）

1. **修复 `train_topo_roi.py` 默认 epochs 覆盖配置问题**（默认应为 `None`，仅显式传参时覆盖）。
2. **统一评估入口与数据模式**：`evaluate.py` 接入 `get_combined_loaders(config)`，并显式区分 DRIVE/Kaggle。
3. **禁止对无 GT 的 split 输出 Dice/IoU**：`split=test` 时只允许推理保存，不输出监督指标。
4. **重构 Kaggle 划分策略**：从 `Training` 划分 train/val，保留 `Test` 仅最终一次评估。
5. **拓扑指标异常值统一处理**：禁止 -1/0 静默混均值；异常应计数并单独报告。
6. **统一拓扑指标定义与日志口径**：明确 CL-Break 是“预测碎片数”还是“与GT差异”。
7. **给 `_find_mask` 增加一一映射校验**：启动时输出冲突并 fail-fast。
8. **补齐/移除 clDice 口径**：实现并纳入 pipeline，或从报告中删除该指标。
9. **对 topology loss 做单样本可解释性单测**：验证 finite/infinite 分量、惩罚方向与期望一致。
10. **冻结实验协议**：统一 threshold、ROI 应用顺序、early-stop 口径，禁止跨脚本混比。

---

## 最小化运行验证记录（受环境限制）

- 已尝试运行基于数据目录的映射核验脚本；当前环境缺少数据目录 `data/combined`，无法完成真实样本级验证。  
- 已尝试导入项目模块进行动态验证；当前环境缺少 `numpy` 等依赖，动态验证受限。

