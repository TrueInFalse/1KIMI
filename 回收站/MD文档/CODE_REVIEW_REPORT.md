# Baseline vs Topo 训练链路内部一致性审计报告

> 审计范围：`train_baseline.py`、`train_with_topology.py`、`topology_loss.py`、`data_combined.py`、`utils_metrics.py`、`config.yaml`（补充查看 `model_unet.py` 与 `evaluate.py` 用于对齐训练/评估口径）。
> 
> 审计方式：只读代码，不改动训练实现；仅依据代码事实给出结论。

---

## 2.1 Executive Summary（高风险点 Top 5）

1. **Topo 分支只在拓扑损失中使用 ROI，Dice 训练损失不使用 ROI（与评估口径不一致）**  
   - 证据：`train_with_topology.py` 中 `loss_dice = DiceLoss(outputs, vessels)` 未传 ROI；`loss_topo = criterion_topo(pred, rois, ...)` 传入 ROI；验证阶段 `compute_basic_metrics/compute_topology_metrics` 都在 ROI 内计算。  
   - 位置：`train_with_topology.py` `train_epoch` 与 `validate`；`utils_metrics.py` 的 ROI 应用逻辑。  
   - 可能后果：训练目标中 Dice 会惩罚 ROI 外区域，而评估时 ROI 外被忽略，优化目标与评估目标偏移，容易出现“训练看似在学，验证 Dice/拓扑不增反降”。  
   - 验证方式（不改代码）：开启 `debug_topo_roi`，并离线对同一 batch 复算“全图 Dice loss vs ROI 内 Dice loss”差值分布。

2. **TopoLoss 的 MSE 模式把 lifetime 显式拉向 0.5，存在“过度平滑/涂抹”风险；且 `max_death` 参数未参与计算**  
   - 证据：`topology_loss.py` 在 `loss_mode='mse'` 时使用 `mse(lifetimes, 0.5)`；构造函数保存了 `max_death`，但 `forward` 未使用该参数。  
   - 位置：`topology_loss.py` `__init__`、`forward`。  
   - 可能后果：若真实拓扑最优 lifetime 分布并不围绕 0.5，MSE 会把所有选中条形向同一目标拉齐，可能牺牲细血管细节；同时配置里改 `max_death` 实际无效，增加实验解释偏差。  
   - 验证方式：读取训练日志中 `topo_loss_raw` 与 `ratio`，对比 `loss_mode=mse` vs `hinge` 的 val_dice/CL-Break 轨迹。

3. **Topo 与 Baseline 日志字段口径不同，易造成“看起来对齐、实际不可比”**  
   - 证据：Baseline CSV 为 `training_log.csv`（无 Dice/Topo 分解字段）；Topo CSV 为 `training_topo_log.csv`（含 `dice_loss/topo_loss_raw/topo_loss_scaled/ratio/roi_mode`）。字段命名也不同（`val_cl_break` vs `cl_break`，`val_delta_beta0` vs `delta_beta0`）。  
   - 位置：`train_baseline.py::_init_log`、`train_with_topology.py::_init_log`。  
   - 可能后果：对齐绘图或表格时容易错列、错口径，造成“Topo 退化”的误读或漏读。  
   - 验证方式：用同一解析脚本读取两种 CSV，先做列名映射再比较，否则禁止直接拼表。

4. **Topo 训练期统计的 `topo_raw/topo_scaled/ratio` 使用 `.item()`，日志值与反向传播图解耦，可能掩盖真实梯度状态**  
   - 证据：`train_with_topology.py` 先 `loss_topo.item()` 计算 raw/scaled/ratio，仅用于日志；真正反传用 `loss = loss_dice + λ*loss_topo`。  
   - 位置：`train_with_topology.py` `train_epoch`。  
   - 可能后果：当出现梯度异常（如某些 batch PH 空集返回 0）时，日志均值可能“平稳”，但参数更新贡献不足。  
   - 验证方式：不改代码下可通过已有日志检查 `topo_loss_raw` 是否长期接近 0，结合 `lambda` 升高后 `val_delta_beta0` 是否仍无变化。

5. **ROI 生成存在“异常回退为全 1”分支，且 train_epoch 的 ROI 全 1 监控只检查每 batch 第一个样本**  
   - 证据：FOV 检测失败时 `_create_fov_mask_from_image` 回退 `ones`；`train_with_topology.py` 只统计 `rois[0,0]` 判断 `roi_all_ones_flag`。  
   - 位置：`data_combined.py::_create_fov_mask_from_image`、`train_with_topology.py::train_epoch`。  
   - 可能后果：部分样本 ROI 异常全 1 会削弱 Topo 对 FOV 外黑边的约束；但由于监控采样不足，问题可能被漏检。  
   - 验证方式：离线遍历一轮 DataLoader，统计每样本 `roi.mean` 与 `unique` 分布，而非只看 batch 首样本。

---

## 2.2 详细审计条目（逐文件）

## 1) `train_baseline.py`

- **该文件负责的功能**  
  Baseline U-Net 训练：模型初始化、Dice loss 训练、验证指标计算、早停、CSV 记录、best checkpoint 保存。

- **与对照链路可能不一致的点**
  1. 训练损失只用 Dice，且未使用 ROI 掩码（全图优化）。
  2. 验证指标在 ROI 内计算（通过 `compute_basic_metrics`/`compute_topology_metrics`）。
  3. CSV 字段是简版，不含 loss 分解信息。

- **可能导致 Topo 欠拟合/伤 Dice 的点（按风险排序）**
  1. 基线自身“训练全图、评估 ROI 内”的口径偏移会作为对照噪声，影响对 Topo 增益/退化的归因。  
  2. 日志无法拆解损失成分，和 Topo 结果做根因对照时信息不足。  
  3. 早停与 best 保存都基于 `val_dice`，与 Topo 一致（此项本身无差异）。

## 2) `train_with_topology.py`

- **该文件负责的功能**  
  Topo 训练总链路：Dice+Topo 联合损失、λ 课程学习、梯度裁剪、验证、日志、checkpoint/early stop。

- **与对照链路可能不一致的点**
  1. 多了 `lambda_scheduler` 与 `criterion_topo`。  
  2. 训练时 Dice 仍是全图；Topo 额外使用 ROI。  
  3. 日志字段更复杂且文件名不同。  
  4. 设备选择与 baseline 不同（此文件固定 `cuda available ? cuda : cpu`，未直接使用 config.device）。

- **可能导致 Topo 欠拟合/伤 Dice 的点（按风险排序）**
  1. 联合损失内 Dice/Topo 的监督区域不一致（Dice 全图，Topo ROI 内）。  
  2. `loss_scale=100`（默认配置）叠加 λ 最高 0.5，Topo 影响可能迅速放大；尽管有 ratio 监控，但没有硬约束。  
  3. `roi_all_ones` 检测粒度不足，可能漏掉局部数据问题。

## 3) `topology_loss.py`

- **该文件负责的功能**  
  计算可微 PH 的 0 维拓扑损失（MSE 或 Hinge），并做 `loss_scale` 缩放。

- **与对照链路可能不一致的点**
  1. 输入是 sigmoid 后概率图（非 logits）。  
  2. ROI 在拓扑分支内以置零方式应用到 `prob`。  
  3. `max_death` 参数未实际参与 forward。

- **可能导致 Topo 欠拟合/伤 Dice 的点（按风险排序）**
  1. MSE 模式将 lifetimes 对齐到固定 0.5，可能与真实血管拓扑分布不匹配。  
  2. PH 空集时直接记 0 loss，可能导致部分 batch 拓扑梯度完全缺失。  
  3. 配置可调项 `max_death` 形同无效，实验可解释性风险高。

## 4) `data_combined.py`

- **该文件负责的功能**  
  Kaggle 联合数据加载、mask 匹配、resize/pad、FOV/ones/tiny ROI 生成、增强与返回 `(image, mask, roi)`。

- **与对照链路可能不一致的点**
  1. ROI 语义在该文件定义（fov/ones/tiny），但训练与评估各模块对 ROI 的使用范围不同。  
  2. FOV 失败时回退全 1，语义接近 `ones` 模式。

- **可能导致 Topo 欠拟合/伤 Dice 的点（按风险排序）**
  1. 若 FOV 估计在某些域（HRF/CHASE/STARE）失效，会弱化 ROI 约束。  
  2. 虽然 resize/pad 对 mask 用 NEAREST，增强后对 roi 也二值化；但 ROI 质量仍取决于前置 FOV 分割鲁棒性。  
  3. 训练中 ROI 参与拓扑而不参与 Dice，造成监督区域错位。

## 5) `utils_metrics.py`

- **该文件负责的功能**  
  统一计算基础分割指标与拓扑指标（CL-Break、Δβ₀），支持 ROI 掩码。

- **与对照链路可能不一致的点**
  1. 评估默认先阈值化再 ROI 裁剪（基础指标通过 `apply_roi_mask` 对向量化样本计算）。  
  2. 拓扑指标采用 `compute_betti0_filtered(min_size=20)`，与训练 Topo loss（基于 lifetime top-k）不是同一目标函数。

- **可能导致 Topo 欠拟合/伤 Dice 的点（按风险排序）**
  1. 训练优化目标与评估拓扑指标并非同构，可能出现“Topo loss 下降但 Δβ₀ 不降”。  
  2. `compute_topology_metrics` 异常返回 -1 逻辑在上层若未识别，可能污染均值解释（Topo 训练验证分支有 try/except 默认 0）。

## 6) `config.yaml`

- **该文件负责的功能**  
  统一定义数据模式、ROI 模式、训练参数、模型参数、指标阈值与拓扑超参。

- **与对照链路可能不一致的点**
  1. `enable_early_stopping: false` 会让两条链路都跑满轮次（一致）。  
  2. `topology.loss_scale: 100.0` + `lambda_schedule: 3175`（warmup=0.1，final=0.5）使 Topo 分支强度较高。  
  3. `topology.max_death` 已配置但当前实现未使用。

- **可能导致 Topo 欠拟合/伤 Dice 的点（按风险排序）**
  1. 高 `loss_scale` 与高 λ 组合若未经 ratio 约束，可能压制 Dice 主目标。  
  2. 无效配置项（`max_death`）增加实验认知偏差。  
  3. ROI 模式设为 `fov`，其质量直接影响 Topo 有效监督区域。

---

## 1.1 Baseline vs Topo 口径一致性对照表

| 对照项 | Baseline | Topo | 一致性 | 影响方向（基于代码事实） |
|---|---|---|---|---|
| 模型构建 | `get_unet_model(in_channels=3, encoder, pretrained, activation)` | 同样调用 `get_unet_model`，参数来源一致 | 一致 | 模型结构本身不是差异源 |
| 预训练加载路径 | 通过 `model_unet.py` 默认 `./pretrained_weights/resnet34-333f7ec4.pth` | 同上 | 一致 | 预训练来源一致 |
| DiceLoss 实现 | `smp.losses.DiceLoss(mode='binary', from_logits=True)` | 同实现同参数 | 一致 | Dice 公式实现一致 |
| 拓扑损失 | 无 | `CubicalRipserLoss`（prob 输入） | 不一致（预期） | 增加额外优化目标 |
| Optimizer | Adam(lr=config) | Adam(lr=config) | 一致 | 不是主要差异 |
| Scheduler | CosineAnnealingLR，同 `T_max/eta_min` | 同 | 一致 | 学习率调度一致 |
| AMP/GradScaler | 未启用 | 未启用 | 一致 | 不是差异源 |
| 梯度裁剪 | 无 | `clip_grad_norm_(1.0)` | 不一致 | Topo 分支更新被额外限幅 |
| Early stop | 同开关、同指标（val_dice） | 同（复用 EarlyStopping） | 一致 | 保存规则对齐 |
| best model 命名 | `best_model.pth` | `best_model_topo.pth` | 不一致（文件名） | 管理层差异，不影响训练 |
| eval阈值 | 使用 `metrics.topology_threshold` | validate 走 `compute_basic_metrics` 默认0.5（未显式传） | 基本一致（当前 config 为0.5） | 若未来配置阈值≠0.5，会出现链路差异 |
| eval中ROI应用 | 指标函数内 ROI 约束 | 同 | 一致 | 指标计算口径一致 |
| 训练中ROI用于 Dice | 否 | 否 | 一致 | 两者均是“训练全图/评估ROI内” |
| 训练中ROI用于 Topo | 不涉及 | 是（仅 Topo） | 不一致 | Topo 分支出现 ROI 专属约束 |

---

## 1.2 TopoLoss 数学语义与实现一致性审计

### 问题逐项回答（基于代码）

1. **topo 输入是 logits 还是 prob？**  
   训练端先 `pred = sigmoid(outputs)`，再调用 `criterion_topo(pred, rois, epoch)`；即输入为 **prob**。

2. **filtration 定义？是否 clamp / detach？**  
   `filtration = 1 - prob`；未看到 clamp 到 [0,1]（但 prob 本身来自 sigmoid）；未 detach。

3. **ROI 外置零在哪里做？影响 Dice 吗？**  
   在 `topology_loss.py` 内部：`prob = prob*roi + (1-roi)*0`，只影响 Topo 分支；Dice 仍用原始 logits 全图计算。

4. **`compute_ph_torch` 返回 `pd` 如何筛选 finite？空集怎么处理？**  
   先筛 `dim==0`，取 birth/death，再 `finite_mask = isfinite(deaths)`；若 finite 数量为 0，则该样本损失记 0。

5. **`topk` 与 `target_beta0` 语义？条形少于k怎么处理？**  
   `target_beta0` 表示最多保留前 k 个最长 lifetime；若条形数 ≤ k，则不做 topk，全部参与损失。

6. **loss 形状（hinge/mse）？是否有硬拉0.5风险？**  
   - hinge: `relu(0.5 - lifetime)^2` 的均值，只惩罚短条形。  
   - mse: `mse(lifetime, 0.5)`，双向拉向 0.5，存在“硬拉 0.5”风险（代码事实）。

7. **数值尺度：loss_scale/lambda/ratio 计算点？是否混用风险？**  
   `TopologicalRegularizer` 返回值已乘 `loss_scale`；训练中 total loss 再乘 λ。日志中 `topo_raw = loss_topo.item()/loss_scale`、`topo_scaled = loss_topo.item()*λ`。反传用张量 `loss_topo`，日志用 `.item()` 标量。

8. **梯度路径是否被 no_grad/numpy/detach 破坏？**  
   Topo 主路径（`sigmoid -> criterion_topo -> total loss`）未见 detach/numpy；可微路径存在。验证阶段有 numpy 转换但在 `@torch.no_grad()` 下，仅评估。

### TopoLoss 计算图流程图（文字版）

`logits(outputs)`  
→ `prob = sigmoid(logits)`  
→ （Topo分支）`prob_roi = prob * roi`（ROI外置0）  
→ `filtration = 1 - prob_roi`  
→ `pd = cripser.compute_ph_torch(filtration, maxdim=0)`  
→ 取 `dim0` 的 `(birth, death)`  
→ 过滤 `death` 非有限值  
→ `lifetime = death - birth`  
→ 若 `len(lifetime)>target_beta0`，取 top-k longest  
→ `loss_i = mse(lifetime,0.5)` 或 `relu(0.5-lifetime)^2`  
→ batch mean  
→ `* loss_scale` 得 `loss_topo`  
→ `total_loss = dice_loss + lambda * loss_topo`  
→ 反向传播

**风险标注：**
- R1：`mse` 模式将 lifetime 拉向固定 0.5。  
- R2：finite 空集直接 0，导致样本无 topo 梯度。  
- R3：`max_death` 未实际生效（配置/实现不一致）。

---

## 1.3 Kaggle ROI（fov/ones/tiny）生成与使用一致性

### 生成逻辑审计

- **确定性**：FOV 生成流程使用阈值、连通域、形态学与椭圆拟合，未使用随机数；因此同一输入图像下是确定性的。随机仅出现在 `_augment`（翻转/旋转），会同步作用于 image/mask/roi。  
- **多数据源边界鲁棒性**：采用 `gray=max(RGB)` + 最大连通域 + 边界点 PCA 椭圆 + dilation 限制，理论上针对黑边有防护；但若前景过少会回退全 1。  
- **插值方式**：mask resize 用 `NEAREST`，旋转增强时 mask/roi 也用 `NEAREST`，最后 `roi_mask=(roi_mask>0.5).float()`，可避免灰度 ROI。  

### ROI 使用位置清单

1. **训练 Dice loss**：未使用 ROI（Baseline/Topo 都是全图 Dice）。  
2. **训练 Topo loss**：仅 Topo 训练使用 ROI（`criterion_topo(pred, rois, ...)`）。  
3. **验证基础指标**：使用 ROI（`compute_basic_metrics`）。  
4. **验证拓扑指标**：使用 ROI（`compute_topology_metrics`）。  
5. **可视化（evaluate.py）**：先 `pred*roi` 再阈值/骨架化。

### 同名变量语义差异

- `roi` 在数据层是空间掩码；在日志层 `roi_mode/roi_mean/roi_all_ones` 是统计摘要。  
- `topology_threshold` 在 baseline validate 显式读取配置；topo validate 依赖 metrics 函数默认阈值 0.5（当前配置恰好一致）。

---

## 1.4 记录与可审计性检查

1. **CSV 字段一致性**：不一致（Topo 多出损失分解和 ROI 字段，列名也不完全同名）。
2. **append/覆盖风险**：两者都在启动时重写日志文件（baseline `csv.writer` 写表头；topo `_init_log(overwrite=True)`）；单次 run 内追加正常。  
3. **best epoch 定义与写入**：两者都按 `val_dice` 最大保存 best；但 Topo 结束打印不显示 best epoch，仅显示 best val_dice。
4. **“看上去对齐、实际口径不对齐”风险**：
   - 直接比两个 CSV 可能列错位；
   - Topo 验证阈值目前依赖默认值，若未来 config 改阈值会静默失配；
   - Topo 训练日志中的 `ratio` 来自 `.item()` 标量，反映数值量级而非真实梯度贡献。

---

## 2.3 验证建议（不改代码，最小成本）

1. **动作A：同一 batch 复算 Dice 口径差（全图 vs ROI 内）**  
   - 目的：验证“训练目标与评估目标偏移”是否显著。  
   - 预期观察：若问题成立，二者差值在包含大黑边样本时显著增大。  
   - 下一步判定：若差值稳定偏大，应优先评估“Dice是否也需ROI约束”这一口径问题。

2. **动作B：对现有 topo 日志做 `ratio`-`val_dice/Δβ₀` 相关性检查**  
   - 目的：验证拓扑项权重是否压制主任务或无效。  
   - 预期观察：若问题成立，`ratio` 升高阶段出现 val_dice 下滑或 `delta_beta0` 无改善。  
   - 下一步判定：若发现拐点，优先做 `loss_mode`（mse/hinge）与 λ 策略的对照复跑。

3. **动作C：离线统计一轮 ROI 质量（每样本）**  
   - 目的：确认 FOV ROI 在联合数据集各子域的稳定性。  
   - 预期观察：若问题成立，会看到一批样本 `roi_mean≈1`（退化全1）或异常偏小。  
   - 下一步判定：若异常集中在某数据源，需单独分域审计 FOV 生成输入质量（黑边/曝光/尺寸）。

---

## 附：关键事实速查（针对必查清单）

- 模型构建：Baseline 与 Topo 都通过 `get_unet_model(in_channels=3, encoder, pretrained, activation)`。  
- DiceLoss：两者均 `smp.losses.DiceLoss(mode='binary', from_logits=True)`。  
- Optimizer/Scheduler：两者均 Adam + CosineAnnealingLR（同参数来源）。  
- AMP：两者都未实现 AMP/GradScaler。  
- Early stop：两者都受 `enable_early_stopping` 控制，best 均按 `val_dice` 保存。  
- Eval 阈值与 ROI 顺序：指标函数中先阈值后 ROI 过滤；可视化中先 ROI 后阈值（仅可视化，不是训练/验证主指标）。

