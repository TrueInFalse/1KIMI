# Codex Betti0 审计修改 — 行为与输出一致性分析

## 总体结论

**Codex 的声明与实际行为 大体一致，但存在一个重要问题：数据集中每张图像被重复处理了两次（`__02` 后缀），导致 40 样本实际只有 20 张独立图像。** 这不影响均值数值的正确性（因为完全重复），但"40 张可视化"的说法具有误导性。

---

## 1. 文件存在性验证 ✅ 全部通过

| 声称的文件/目录 | 是否存在 | 备注 |
|---|---|---|
| `experiments/audit_betti0_per_image.py` | ✅ | 25,900 字节，668 行 |
| `reports/BETTI0_AUDIT_REPORT.md` | ✅ | 1,199 字节 |
| `reports/audits/betti0_gt_val_per_image.csv` | ✅ | 40 行数据 + 表头 |
| `reports/audits/betti0_baseline_val_per_image.csv` | ✅ | 40 行数据 + 表头 |
| `reports/audits/betti0_topo_lcap020_val_per_image.csv` | ✅ | 40 行数据 + 表头 |
| `results/betti0_audit/gt_components/` | ✅ | 40 张 PNG |
| `results/betti0_audit/baseline_components/` | ✅ | 40 张 PNG |
| `results/betti0_audit/topo_lcap020_components/` | ✅ | 40 张 PNG |

---

## 2. 脚本功能验证 ✅ 基本通过

### CLI 参数声明

| 声称的参数 | 代码中是否存在 | 实际默认值 |
|---|---|---|
| `--config` | ✅ | `config_125e.yaml` |
| `--split` | ✅ | `val` |
| `--checkpoint` | ✅ | `None` |
| `--label` | ✅ | `None` |
| `--threshold` | ✅ | `0.5` |
| `--min-sizes` | ✅ | `[5, 10, 20, 50]` |
| `--max-samples` | ✅ | `None` |
| `--output-dir` | ✅ | `reports/audits` |

### 功能声明核查

| 声称的功能 | 代码验证 |
|---|---|
| 复用 `get_combined_loaders` + `load_model` | ✅ 第 39-40 行 import，第 421/437 行调用 |
| ROI 口径后再统计 Betti0 | ✅ 第 473 行 `gt_binary = gt_binary * roi_binary` |
| 支持 GT-only 与 checkpoint 审计 | ✅ 第 433-440 行分支逻辑 |
| 导出 per-image raw/filtered Betti0 | ✅ CSV 字段完整 |
| 支持多 `min_size` | ✅ 第 56-60 行 + 第 487-488 行 |
| 导出 GT skeleton fragments | ✅ 第 476-477 行 |
| 生成连通域着色图 | ✅ 第 152-163 行 `colorize_components` |
| 自动更新 Markdown 报告 | ✅ 第 276-395 行 `update_audit_report` |
| `num_workers=0` 仅该脚本 | ✅ 第 419 行，运行时深拷贝 config（第 411 行） |

---

## 3. 主线文件完整性验证 ✅ 通过

```
git diff HEAD~1 -- utils_metrics.py train_topo_roi.py evaluate.py
→ 空输出（无改动）

git diff HEAD~1 -- loss_topo*.py losses/ loss*.py
→ 空输出（无改动）
```

> [!NOTE]
> Codex 声称"未改动主线文件"，经 git diff 验证确认 `utils_metrics.py`、`train_topo_roi.py`、`evaluate.py`、loss 相关文件均无变更。

---

## 4. 数值声明验证

### GT 统计

| 声称的数值 | CSV 实际计算值 | 是否一致 |
|---|---|---|
| `gt_beta0_raw` 均值 = **985.25** | **985.25** | ✅ |
| `gt_beta0_filtered_ms5` 均值 = **149.7** | **149.70** | ✅ |
| `gt_beta0_filtered_ms10` 均值 未声称 | **57.00** | — |
| `gt_beta0_filtered_ms20` 均值 = **19.05** | **19.05** | ✅ |
| `gt_beta0_filtered_ms50` 均值 = **6.1** | **6.10** | ✅ |

### Baseline Δβ₀ 均值

| min_size | 声称值 | 实际值 | 一致 |
|---|---|---|---|
| ms5 | 78.3 | **78.30** | ✅ |
| ms10 | 26.65 | **26.65** | ✅ |
| ms20 | 13.7 | **13.70** | ✅ |
| ms50 | 7.85 | **7.85** | ✅ |

### Topo_lcap020 Δβ₀ 均值

| min_size | 声称值 | 实际值 | 一致 |
|---|---|---|---|
| ms5 | 144.7 | **144.70** | ✅ |
| ms10 | 52.65 | **52.65** | ✅ |
| ms20 | 15.6 | **15.60** | ✅ |
| ms50 | 5.2 | **5.20** | ✅ |

### Topo pred_beta0_filtered_ms20 均值

| 声称值 | 实际值 | 一致 |
|---|---|---|
| 3.65 | **3.65** | ✅ |

### 报告中的 Markdown 内容

| 报告声称 | 实际写入 | 一致 |
|---|---|---|
| val 集统计 | ✅ 含 raw 和 4 个 ms 的均值 | ✅ |
| Top5 碎裂样本 | ✅ 35_DRIVE(47), 14_h_HRF(44), 15_h_HRF(37) | ✅ |
| baseline/topo 对比 | ✅ 含 pred_mean 和 delta_mean | ✅ |
| 一句话解释 | ✅ 4 条解读 | ✅ |

> [!TIP]
> 所有数值声明与 CSV 实际数据 **完全一致**，精度匹配到小数点后 2 位。

---

## 5. 发现的问题

### ⚠️ 重大问题：样本重复（20 独立图 × 2 = 40 行）

> [!WARNING]
> **40 个样本实际只有 20 张独立图像。** 每张图像被 DataLoader 产出了两次，脚本将第二次命名为 `{id}__02`。

**证据：**
- CSV 中 `base_image_id` 只有 20 个唯一值
- 每个 `__02` 行的所有数值字段与原始行 **完全相同**
- 可视化图片 MD5 哈希完全一致（`13_dr_HRF.png` = `13_dr_HRF__02.png`）
- 图片文件大小逐对完全一致

**原因分析：**
- `get_combined_loaders` 的 val 数据集可能对每张图做了两次 crop/patch（例如双通道标注、两种 augmentation）
- 但由于 retinal 数据集的 ROI 和 GT 在同一张图上是固定的，两次产出的内容完全一样
- 脚本通过 `name_counts` 机制正确识别了重复并加上 `__02` 后缀，说明知道会有重复，但未去重

**影响：**
- **均值不受影响**：完全重复的行不改变平均值
- **"40 张可视化"有误导性**：实际只有 20 张独立可视化，另外 20 张是像素级完全相同的副本
- **Top5 排名有虚假并列**：`35_DRIVE` 和 `35_DRIVE__02` 占了两个名额，本质是同一张图

### ⚠️ 次要问题：审计文件未提交 Git

> [!IMPORTANT]
> `experiments/audit_betti0_per_image.py` 和 `reports/BETTI0_AUDIT_REPORT.md` 状态为 `??`（untracked）。
> `reports/audits/` 也是 untracked。
> `results/betti0_audit/` 被 `.gitignore` 中的 `results/` 规则忽略。
>
> Codex 声称"修改了仓库"，但这些文件实际上从未被 commit。这可能是 Codex 运行环境的限制（只写文件不 commit），但需要你手动 `git add` 并提交。

---

## 6. 行为与输出一致性评定

| 维度 | 评定 | 说明 |
|---|---|---|
| 文件是否全部生成 | ✅ 完全一致 | 所有声称的文件/目录均存在 |
| 脚本功能是否与描述匹配 | ✅ 完全一致 | CLI、import、ROI 口径、num_workers 等全部验证通过 |
| 数值是否与 CSV 数据一致 | ✅ 完全一致 | 所有 12 个数值声明均精确匹配 |
| 主线文件是否未改动 | ✅ 完全一致 | git diff 确认零改动 |
| 报告内容是否与声明匹配 | ✅ 完全一致 | Top5、对比表、解释句全部在位 |
| 可视化数量 "各 40 张" | ⚠️ 名义一致，实质存疑 | 确实有 40 个文件，但其中 20 个是完全重复的副本 |
| Git 提交 | ❌ 未提交 | 核心脚本和报告都是 untracked 状态 |

---

## 7. 建议操作

1. **去重或标注**：在后续使用时，应基于 20 个独立样本报告统计，或在论文中注明含重复
2. **Git 提交**：`git add experiments/audit_betti0_per_image.py reports/BETTI0_AUDIT_REPORT.md reports/audits/` 后 commit
3. **排查重复源**：检查 `get_combined_loaders` 的 val split 为何产出 40 而非 20 条样本（可能是数据集的双标注通道导致）
