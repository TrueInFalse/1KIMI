# evaluate.py 模块说明

## 功能概述

`evaluate.py`是模型评估脚本，实现：

- **指标计算**：Dice、IoU、Precision、Recall、CL-Break、Δβ₀（均在ROI内）
- **可视化**：原图、GT、预测、骨架对比
- **预测保存**：保存预测概率图（.npy格式）

## 接口定义（Input/Output）

### evaluate_model函数

```python
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda',
    compute_topology: bool = True,
    save_predictions: bool = False,
    output_dir: Optional[Path] = None
) -> Dict[str, float]
```

**返回**：包含所有平均指标的字典

### visualize_results函数

```python
def visualize_results(
    model: torch.nn.Module,
    dataset: DRIVEDataset,
    device: str = 'cuda',
    num_samples: int = 4,
    save_path: Optional[str] = None
) -> None
```

**输出**：每行5张图（原图、GT血管、预测血管、GT骨架、预测骨架）

## 使用示例

```bash
# 评估验证集（默认）
python evaluate.py

# 评估训练集
python evaluate.py --split train

# 指定检查点
python evaluate.py --checkpoint checkpoints/best_model.pth

# 评估测试集
python evaluate.py --split test
```

## 输出文件

```
results/
├── evaluation_results.png      # 可视化结果
└── predictions/
    ├── 21_pred.npy            # 预测概率图（可选）
    ├── 22_pred.npy
    └── ...
```

## 可视化说明

每行包含5张图像：

| 列 | 内容 | 说明 |
|----|------|------|
| 1 | 原图（绿色通道） | 眼底原始图像 |
| 2 | GT血管 | 人工标注的血管标签 |
| 3 | 预测血管 | 模型输出（概率图） |
| 4 | GT骨架 | 人工标注的骨架 |
| 5 | 预测骨架 | 预测结果的骨架化 |

**诊断技巧**：
- 对比GT和预测骨架：碎片越多说明断裂越严重
- CL-Break高但Dice不低：模型捕获了血管位置但连通性差

## 依赖清单

**Python包**：
- `torch`: PyTorch框架
- `skimage`: 骨架化
- `matplotlib`: 可视化
- `tqdm`: 进度条

## 更新日志

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v1.0 | 2026-02-07 | 初始版本，实现评估和可视化 |
