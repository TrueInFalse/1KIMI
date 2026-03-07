# train_baseline.py 模块说明

## 功能概述

`train_baseline.py`是U-Net基线训练脚本，实现：

- **训练循环**：带进度条的epoch训练
- **验证循环**：计算所有指标（含拓扑指标）
- **早停机制**：仅监控`val_dice`，拓扑指标不触发早停
- **学习率调度**：CosineAnnealingLR
- **日志记录**：CSV + 控制台（指定格式）

## 接口定义（Input/Output）

### Trainer类

```python
class Trainer:
    def __init__(self, config: Dict, device: str = 'cuda') -> None
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None
```

### main函数

```python
def main(config_path: str = 'config.yaml') -> None
```

## 使用示例

```bash
# 直接运行（默认使用config.yaml）
python train_baseline.py

# 指定配置文件
python train_baseline.py --config my_config.yaml
```

**代码中使用**：
```python
from train_baseline import Trainer

trainer = Trainer(config, device='cuda')
trainer.train(train_loader, val_loader)
```

## 日志格式

### 控制台输出

```
Epoch   1 | Train Dice: 0.2341 | Val Dice: 0.4523 | Val CL-Break: 125.3 | Val Δβ₀: 12.5 | LR: 0.000100
Epoch   2 | Train Dice: 0.4567 | Val Dice: 0.5634 | Val CL-Break:  98.2 | Val Δβ₀:  8.3 | LR: 0.000098
...
```

### CSV日志（logs/training_log.csv）

```csv
epoch,train_loss,train_dice,val_dice,val_iou,val_precision,val_recall,val_cl_break,val_delta_beta0,lr
1,0.5432,0.2341,0.4523,0.3245,0.6123,0.4123,125.3,12.5,0.000100
...
```

## 关键设计说明

### 拓扑指标仅用于日志

```python
# 早停仅基于val_dice
self.early_stopping = EarlyStopping(patience=20, mode='max')

# 拓扑指标在validate()中计算，但不影响早停
val_metrics = self.validate(val_loader)
if self.early_stopping(val_metrics['dice'], epoch):
    # 仅当val_dice无改善时才触发早停
```

### 损失函数配置

```python
self.criterion = smp.losses.DiceLoss(
    mode='binary',
    from_logits=True  # 模型输出logits，内部自动sigmoid
)
```

注意：模型输出的是logits（无sigmoid），DiceLoss内部会自动应用sigmoid。

### 学习率调度

```python
self.scheduler = CosineAnnealingLR(
    self.optimizer,
    T_max=max_epochs,
    eta_min=lr * 0.01
)
```

学习率从初始值下降到初始值的1%。

## 输出文件

```
./
├── checkpoints/
│   └── best_model.pth          # 最佳模型（仅保存一个）
└── logs/
    └── training_log.csv        # 训练日志
```

## 依赖清单

**Python包**：
- `torch`: PyTorch框架
- `segmentation-models-pytorch`: 损失函数
- `tqdm`: 进度条
- `yaml`: 配置解析

## 更新日志

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v1.0 | 2026-02-07 | 初始版本，实现基础训练流程 |
| v1.1 | 2026-02-07 | 添加拓扑指标日志（CL-Break、Δβ₀），仅监控不触发早停 |
| v1.2 | 2026-02-07 | 增强日志输出：显示所有指标、每轮用时、总用时、开始/结束时间 |
