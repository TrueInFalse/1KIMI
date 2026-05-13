# Betti0 审计去重修复 — 最终审计报告（第二轮）

**审计日期**: 2026-04-16  
**审计对象**: `experiments/audit_betti0_per_image.py` 去重补丁及其全部产出  
**审计范围**: CSV / 报告 / 可视化 / 主线污染  
**前置**: 第一轮审计发现旧可视化目录未清理，Codex 已完成清理后进入本轮复审。

---

## 1. 审计总评

> **通过：重复样本问题已修复，Betti0 审计结果现在可用于论文。**

---

## 2. 分项结论

### A. 重复样本是否真的被修掉

| 检查项 | 结论 | 证据 |
|--------|------|------|
| GT CSV 重复 | ✅ 通过 | `betti0_gt_val_per_image.csv` = 20 行数据，20 个独立 `image_id`，无 `__02` |
| baseline CSV 重复 | ✅ 通过 | `betti0_baseline_val_per_image.csv` = 20 行数据，20 个独立 `image_id`，无 `__02` |
| topo CSV 重复 | ✅ 通过 | `betti0_topo_lcap020_val_per_image.csv` = 20 行数据，20 个独立 `image_id`，无 `__02` |
| 独立样本数 | ✅ 通过 | 报告写 `Independent samples: 20`，与 CSV 行数一致 |
| `__02` 条目 | ✅ 通过 | 三个 CSV 中 `image_id` 列均无 `__02` 后缀 |

**结论**: 通过  
**风险等级**: 低

---

### B. 去重是否安全

| 检查项 | 结论 | 证据 |
|--------|------|------|
| 去重依据 | `base_image_id` | 脚本 L584: `base_image_id = sample_name(dataset, ...)` |
| 依据稳定性 | ✅ 合理 | 使用 `dataset.image_ids`（int→格式化）或 `dataset.image_files`（stem），确定性可复现 |
| 不一致报错 | ✅ 是 | `assert_duplicate_rows_consistent()` (L248-266) 逐字段比较 GT/Pred 行；`assert_duplicate_visual_signature_consistent()` (L269-284) 用 SHA1 校验像素级数据，任何不一致直接 `RuntimeError` |
| 粗暴丢弃风险 | ✅ 无 | 首次出现写入字典，后续重复仅做一致性校验后丢弃，不覆盖不合并（L662-706） |

**结论**: 通过  
**风险等级**: 低

---

### C. 去重后数值是否仍正确

| 检查项 | 结论 | 验算 |
|--------|------|------|
| GT 均值 `ms20` | ✅ 正确 | CSV 20 行 ms20 值: 12+14+26+10+6+44+17+12+37+23+33+47+36+27+4+3+7+5+11+7 = **381**, 381÷20 = **19.05** ✓ 报告写 19.0500 |
| baseline 均值 `ms20` | ✅ 正确 | CSV pred_ms20: 18+37+32+41+26+29+40+20+30+14+12+24+31+24+15+14+26+20+19+17 = **489**, 489÷20 = **24.45** ✓ 报告写 24.4500 |
| topo 均值 `ms20` | ✅ 正确 | CSV pred_ms20: 2+7+3+3+4+2+4+1+1+2+2+2+6+2+4+4+8+5+8+3 = **73**, 73÷20 = **3.65** ✓ 报告写 3.6500 |
| Top5 去重 | ✅ 通过 | 5 个不同 image_id: `35_DRIVE(47), 14_h_HRF(44), 15_h_HRF(37), 36_DRIVE(36), 33_DRIVE(33)`，无重复占位 |
| 报告与 CSV 一致性 | ✅ 通过 | 报告 CSV rows / visuals / 均值 / Top5 均与实际 CSV 数据吻合 |

**结论**: 通过  
**风险等级**: 低

---

### D. 可视化目录是否已与独立样本数一致

| 目录 | 文件数 | `__02` 文件 | 结论 |
|------|--------|------------|------|
| `results/betti0_audit/`（旧） | **0 文件** | 无 | ✅ 已清空 |
| `results/betti0_audit_unique/gt_components_independent/` | **20** | 无 | ✅ 干净 |
| `results/betti0_audit_unique/baseline_components_independent/` | **20** | 无 | ✅ 干净 |
| `results/betti0_audit_unique/topo_lcap020_components_independent/` | **20** | 无 | ✅ 干净 |

> [!NOTE]
> `results/betti0_audit_unique/` 下有一个空的 `gt_components/` 子目录（0 文件），属于历史残留空目录。不影响结果，可随时删除。

**结论**: 通过  
**风险等级**: 低

---

### E. 主线是否仍然未被污染

| 文件 | 改动情况 | 与去重的关系 | 安全性 |
|------|---------|------------|--------|
| `utils_metrics.py` | 有改动（`_component_sizes()` 向量化重构） | **无关** — 纯性能优化，`for` 循环→`np.bincount`，语义等价 | ✅ 安全 |
| `train_topo_roi.py` | 有改动（日志格式/变体路由/`fast_dev`） | **无关** — 训练入口扩展 | ✅ 安全 |
| `evaluate.py` | 无未提交改动（`git diff HEAD` 为空） | 无关 | ✅ 安全 |
| `topology_loss_fragment_suppress.py` | 无改动（`git diff HEAD` 为空） | 无关 | ✅ 安全 |
| loss 数学逻辑 | 未被触及 | — | ✅ 安全 |

**结论**: 通过 — 此次修复完全局限于 `experiments/audit_betti0_per_image.py` 和其输出文件，主线训练/评估/loss 均未受影响。  
**风险等级**: 低

---

## 3. 最值得警惕的点

无高风险项。以下为低优先级备注：

1. **空目录残留**: `results/betti0_audit/`（已清空但目录本身存在）和 `results/betti0_audit_unique/gt_components/`（空目录）。不影响任何结果，建议在下次整理时顺手删除。

---

## 4. 最终一句话判断

> **现在这份 Betti0 审计结果，可以作为论文里解释 19.1 和 Δβ₀ 含义的正式证据使用。**
>
> CSV（3×20行）、报告（均值/Top5/样本数）、可视化（3×20张）三者完全一致，去重逻辑带有一致性校验保护，主线训练/评估/loss 未被触及。无卡点。
