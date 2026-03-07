# visualize_results.py 模块说明

## 功能概述

`visualize_results.py`是可视化展示模块，用于：

- **验证集对比**：展示原图、ROI掩码、金标血管、预测血管
- **测试集预测**：展示原图、ROI掩码、预测血管（测试集无金标）
- **训练曲线**：生成2×3布局的6指标训练曲线图
- **随机采样**：可指定样本索引或随机选择

## 接口定义（Input/Output）

### visualize_val_sample函数

```python
def visualize_val_sample(
    model: torch.nn.Module,
    dataset: DRIVEDataset,
    device: str = 'cuda',
    sample_idx: Optional[int] = None,
    save_path: str = 'results/val_sample_comparison.png'
) -> None
```

**输出**：2×2对比图（原图、ROI掩码、金标、预测）

### visualize_test_sample函数

```python
def visualize_test_sample(
    model: torch.nn.Module,
    dataset: DRIVEDataset,
    device: str = 'cuda',
    sample_idx: Optional[int] = None,
    save_path: str = 'results/test_sample_prediction.png'
) -> None
```

**输出**：1×3对比图（原图、ROI掩码、预测）

### plot_training_curves函数

```python
def plot_training_curves(
    log_file: Path = Path('logs/training_log.csv'),
    save_path: Path = Path('results/training_curves.png')
) -> None
```

**输出**：2×3训练曲线图（Dice、Loss、IoU、CL-Break、Precision、Δβ₀）

## 使用示例

```bash
# 默认使用best_model.pth，随机选择样本
python visualize_results.py

# 指定模型和样本索引
python visualize_results.py --checkpoint checkpoints/best_model.pth --val-idx 2 --test-idx 5

# 仅指定验证集样本
python visualize_results.py --val-idx 0

# 仅生成训练曲线图
python visualize_results.py --plot-curves
```

## 输出文件

```
results/
├── val_sample_comparison.png    # 验证集对比图
├── test_sample_prediction.png   # 测试集预测图
└── training_curves.png          # 训练曲线图（2×3布局）
```

## 可视化布局

### 验证集对比图（2×2布局）

| 位置 | 内容 | 说明 |
|------|------|------|
| 左上 | 原图 | 绿色通道眼底图像 |
| 右上 | ROI掩码 | 圆形眼底区域（覆盖率~68%） |
| 左下 | 金标血管 | 人工标注的血管标签（~8%） |
| 右下 | 预测血管 | 模型输出的概率图 |

### 测试集预测图（1×3布局）

| 位置 | 内容 | 说明 |
|------|------|------|
| 左 | 原图 | 绿色通道眼底图像 |
| 中 | ROI掩码 | 圆形眼底区域 |
| 右 | 预测血管 | 模型输出的概率图 |

### 训练曲线图（2×3布局）

| 行 | 列1 | 列2 | 列3 |
|----|-----|-----|-----|
| 第一行 | Dice系数（Train/Val） | 损失函数（Train） | IoU（Val） |
| 第二行 | CL-Break（Val） | Precision（Val） | Δβ₀（Val） |

## 依赖清单

**Python包**：
- `torch`: PyTorch框架
- `matplotlib`: 可视化
- `yaml`: 配置解析

## 更新日志

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v1.0 | 2026-02-07 | 初始版本，实现对比图可视化 |
| v1.1 | 2026-02-07 | 添加训练曲线生成功能（2×3布局，6个指标） |
