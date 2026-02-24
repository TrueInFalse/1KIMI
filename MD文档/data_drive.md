# data_drive.py 模块说明

## 功能概述

`data_drive.py`是DRIVE视网膜血管分割项目的数据加载模块，负责：

1. **数据集划分**：实现16+4划分策略（训练21-36，验证37-40）
2. **路径管理**：严格区分`1st_manual`（血管标签）和`mask`（ROI）
3. **数值验证**：防止误加载ROI mask导致训练失败
4. **预处理**：绿色通道提取、归一化、resize
5. **数据增强**：仅水平翻转（Stage 0简化策略）

## 接口定义（Input/Output）

### DRIVEDataset类

```python
class DRIVEDataset(Dataset):
    def __init__(
        self,
        root: Union[str, Path],      # 数据集根目录
        image_ids: List[int],         # 图像ID列表 [21, 22, ...]
        img_size: int = 512,          # 输出尺寸（原图565×584，512接近原尺寸且是2的幂次）
        in_channels: int = 1,         # 1=绿色通道, 3=RGB
        is_training: bool = True      # 是否启用数据增强
    ) -> None
```

**输出（__getitem__返回）**：

| 变量名 | 形状 | 值域 | 说明 |
|--------|------|------|------|
| image | [C, H, W] | [0, 1] | 预处理后的图像 |
| vessel_mask | [1, H, W] | {0, 1} | 血管标签（细线状） |
| roi_mask | [1, H, W] | {0, 1} | ROI掩码（圆形区域） |

### get_drive_loaders函数

```python
def get_drive_loaders(
    config_path: str = 'config.yaml'
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]
```

**返回**：`(train_loader, val_loader, test_loader)`

## 使用示例

### 基本用法

```python
from data_drive import DRIVEDataset, get_drive_loaders

# 方法1：直接使用Dataset
dataset = DRIVEDataset(
    root='./DRIVE',
    image_ids=[21, 22, 23, 24],
    img_size=512,
    in_channels=1
)
img, vessel, roi = dataset[0]

# 方法2：使用工厂函数
train_loader, val_loader, test_loader = get_drive_loaders('config.yaml')

for batch in train_loader:
    images, vessels, rois = batch
    # 训练代码...
    break
```

### 验证数据正确性

```python
# 检查血管标签均值（应~0.1，若为~0.5则可能误加载了ROI）
dataset = DRIVEDataset(root='./DRIVE', image_ids=[21], img_size=512)
img, vessel, roi = dataset[0]

print(f'血管标签均值: {vessel.mean():.4f}')  # 期望: 0.05-0.15
print(f'ROI掩码均值: {roi.mean():.4f}')      # 期望: ~0.5
```

## 依赖清单

**Python包**：
- `torch`: 张量操作
- `numpy`: 数组处理
- `PIL`: 图像加载（支持.tif和.gif）
- `yaml`: 配置文件解析

**文件依赖**：
```
config.yaml
└── DRIVE/
    ├── training/
    │   ├── images/21_training.tif ...
    │   ├── 1st_manual/21_manual1.gif ...  # 血管标签
    │   └── mask/21_training_mask.gif ...  # ROI掩码
    └── test/
        ├── images/01_test.tif ...
        └── mask/01_test_mask.gif ...      # 仅ROI，无血管标签
```

## 关键检查点说明

### 1. 血管标签数值范围检查

位置：`__getitem__`方法

```python
vessel_mean = vessel_mask.mean().item()
if not (0.02 < vessel_mean < 0.30):
    warnings.warn('血管标签均值异常，可能误加载了ROI mask')
```

**原理**：
- 血管标签（细线）：占~10%像素，mean≈0.1
- ROI mask（圆形）：占~50%像素，mean≈0.5
- 若mean>0.3，极可能是加载了错误的文件

### 2. 路径验证

位置：`_validate_data`方法

自动检查：
- 图像文件存在性
- 血管标签存在性（训练集）
- ROI掩码存在性

### 3. 掩码二值化

位置：`_load_mask`方法

```python
mask_array = (mask_array > 0).astype(np.float32)
```

确保掩码严格为{0, 1}，避免.gif文件的灰度值干扰。

## 更新日志

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v1.0 | 2026-02-07 | 初始版本，实现基础数据加载 |
| v1.1 | 2026-02-07 | 增加血管标签数值范围验证，防止误加载ROI mask |
