# model_unet.py 模块说明

## 功能概述

`model_unet.py`负责U-Net模型的创建和加载，基于`segmentation-models-pytorch`库：

- **编码器**：ResNet34（ImageNet预训练）
- **输出**：单通道logits（无sigmoid激活）
- **输入**：1通道（绿色通道）或3通道（RGB）

## 接口定义（Input/Output）

### get_unet_model函数

```python
def get_unet_model(
    in_channels: int = 1,
    encoder: str = 'resnet34',
    pretrained: bool = True,
    activation: Optional[str] = None
) -> nn.Module
```

**参数**：
- `in_channels`: 输入通道数（1=绿色通道）
- `encoder`: 编码器名称（'resnet34'）
- `pretrained`: 是否使用ImageNet预训练
- `activation`: 输出激活（None=输出logits）

**输出**：smp.Unet模型实例

### load_model函数

```python
def load_model(
    checkpoint_path: Union[str, Path],
    device: str = 'cuda'
) -> nn.Module
```

**功能**：从检查点加载模型权重

## 使用示例

```python
from model_unet import get_unet_model, load_model

# 创建模型
model = get_unet_model(in_channels=1, encoder='resnet34', pretrained=True)
model = model.to('cuda')

# 前向传播
import torch
x = torch.randn(4, 1, 512, 512).cuda()
y = model(x)  # [4, 1, 512, 512]

# 加载保存的模型
model = load_model('checkpoints/best_model.pth')
```

## 依赖清单

**Python包**：
- `torch`: PyTorch框架
- `segmentation-models-pytorch`: U-Net实现

**文件依赖**：
```
config.yaml
  └── model:
        encoder: resnet34
        pretrained: true
```

## 关键设计说明

### 为什么输出logits而不是probability？

```python
activation=None  # 输出logits
```

配合`smp.losses.DiceLoss(mode='binary', from_logits=True)`使用：
- DiceLoss内部会自动sigmoid + 阈值处理
- 避免double sigmoid问题
- 数值稳定性更好

### 关于in_channels与pretrained的冲突

预训练权重（ImageNet）是3通道的，如果使用`in_channels=1`：
- smp库会自动处理：复制第一个通道的权重，或重新初始化
- 这是预期行为，不影响训练

## 更新日志

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v1.0 | 2026-02-07 | 初始版本，使用smp.Unet + ResNet34 |
