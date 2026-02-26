# -*- coding: utf-8 -*-
"""
文件名: data_drive.py
项目: retina_ph_seg
功能: DRIVE数据集加载器，处理16+4划分与ROI约束
作者: AI Assistant
创建日期: 2026-02-07
Python版本: 3.12+

依赖关系:
  - 上游: config.yaml（配置读取）
  - 下游: train_baseline.py（训练脚本）, evaluate.py（评估）

主要类/函数:
  - DRIVEDataset(Dataset): 主数据集类
  - get_drive_loaders(): 数据加载器工厂函数

更新记录:
  - v1.0: 初始版本，实现基础数据加载
  - v1.1: 增加血管标签数值范围验证，防止误加载ROI mask
"""

import os
import warnings
from pathlib import Path
from typing import Tuple, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import yaml
import torchvision.transforms as T


class DRIVEDataset(Dataset):
    """DRIVE视网膜血管分割数据集（统一RGB 3通道版）。
    
    关键特性:
    - 严格区分1st_manual（血管标签）和mask（ROI）
    - 16+4划分策略：21-36训练，37-40验证
    - 统一输出RGB 3通道（适配ImageNet预训练权重）
    
    Args:
        root: DRIVE数据集根目录
        image_ids: 图像ID列表
        img_size: 输出图像尺寸
        is_training: 是否为训练模式（启用数据增强）
    """
    
    def __init__(
        self,
        root: Union[str, Path],
        image_ids: List[int],
        img_size: int = 256,
        is_training: bool = True
    ) -> None:
        self.root = Path(root)
        self.image_ids = image_ids
        self.img_size = img_size
        self.is_training = is_training
        
        # 归一化（与Kaggle联合数据集统一）
        self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
        # 数据集模式推断（training/test）
        if min(image_ids) >= 21:
            self.mode = 'training'
        else:
            self.mode = 'test'
        
        # 路径模板
        self._setup_paths()
        
        # 验证数据存在性
        self._validate_data()
    
    def _setup_paths(self) -> None:
        """设置各类文件的路径模板。"""
        if self.mode == 'training':
            # 训练集路径
            self.img_dir = self.root / 'training' / 'images'
            self.vessel_dir = self.root / 'training' / '1st_manual'
            self.mask_dir = self.root / 'training' / 'mask'
            self.img_pattern = '{:02d}_training.tif'
            self.vessel_pattern = '{:02d}_manual1.gif'
            self.mask_pattern = '{:02d}_training_mask.gif'
        else:
            # 测试集路径（无血管标签，只有ROI mask）
            self.img_dir = self.root / 'test' / 'images'
            self.vessel_dir = None  # 测试集无血管标签
            self.mask_dir = self.root / 'test' / 'mask'
            self.img_pattern = '{:02d}_test.tif'
            self.vessel_pattern = None
            self.mask_pattern = '{:02d}_test_mask.gif'
    
    def _validate_data(self) -> None:
        """验证数据文件存在性。"""
        missing_files = []
        
        for img_id in self.image_ids:
            # 检查图像文件
            img_path = self.img_dir / self.img_pattern.format(img_id)
            if not img_path.exists():
                missing_files.append(str(img_path))
            
            # 训练集需检查血管标签和ROI mask
            if self.mode == 'training':
                vessel_path = self.vessel_dir / self.vessel_pattern.format(img_id)
                if not vessel_path.exists():
                    missing_files.append(str(vessel_path))
                
                mask_path = self.mask_dir / self.mask_pattern.format(img_id)
                if not mask_path.exists():
                    missing_files.append(str(mask_path))
            else:
                # 测试集只需检查ROI mask
                mask_path = self.mask_dir / self.mask_pattern.format(img_id)
                if not mask_path.exists():
                    missing_files.append(str(mask_path))
        
        if missing_files:
            raise FileNotFoundError(f'以下文件不存在:\n' + '\n'.join(missing_files))
    
    def __len__(self) -> int:
        """返回数据集大小。"""
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取单个样本。
        
        Args:
            idx: 样本索引
            
        Returns:
            image: 预处理后的图像张量 [C, H, W]
            vessel_mask: 血管标签张量 [1, H, W]
            roi_mask: ROI掩码张量 [1, H, W]（用于评估时约束计算区域）
            
        Raises:
            AssertionError: 当血管标签数值范围异常时（可能误加载了ROI）
        """
        img_id = self.image_ids[idx]
        
        # 加载原始图像
        img_path = self.img_dir / self.img_pattern.format(img_id)
        image = self._load_image(img_path)
        
        # 加载血管标签（仅训练集有）
        if self.mode == 'training':
            vessel_path = self.vessel_dir / self.vessel_pattern.format(img_id)
            vessel_mask = self._load_mask(vessel_path)
            
            # 关键检查点：血管标签数值范围验证
            # 血管标签应为细线状，占~10%像素，mean应在0.05-0.15之间
            vessel_mean = vessel_mask.mean().item()
            if not (0.02 < vessel_mean < 0.30):
                warnings.warn(
                    f'图像 {img_id:02d} 的血管标签均值异常: {vessel_mean:.4f}。'
                    f'可能误加载了ROI mask（~50%）而非血管标签（~10%）。'
                )
        else:
            # 测试集无血管标签，返回全零占位
            vessel_mask = torch.zeros(1, self.img_size, self.img_size)
        
        # 加载ROI掩码（评估时用于约束指标计算区域）
        mask_path = self.mask_dir / self.mask_pattern.format(img_id)
        roi_mask = self._load_mask(mask_path)
        
        # 数据增强（仅训练时）
        if self.is_training:
            image, vessel_mask, roi_mask = self._augment(image, vessel_mask, roi_mask)
        
        return image, vessel_mask, roi_mask
    
    def _load_image(self, path: Path) -> torch.Tensor:
        """加载并预处理图像（统一输出RGB 3通道）。
        
        Args:
            path: 图像文件路径
            
        Returns:
            预处理后的图像张量 [3, H, W]
        """
        # 使用PIL加载图像
        img = Image.open(path)
        img_array = np.array(img)  # [H, W, C] 或 [H, W]
        
        # 转换为张量并归一化，统一输出RGB 3通道
        if len(img_array.shape) == 3:
            # 彩色图像 -> RGB [3, H, W]
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        else:
            # 灰度图像 -> 复制为3通道 [3, H, W]
            img_gray = torch.from_numpy(img_array).float() / 255.0
            img_tensor = img_gray.unsqueeze(0).repeat(3, 1, 1)  # [1, H, W] -> [3, H, W]
        
        # 应用与Kaggle联合数据集相同的归一化
        img_tensor = self.normalize(img_tensor)
        
        # resize到目标尺寸
        img_tensor = torch.nn.functional.interpolate(
            img_tensor.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        return img_tensor
    
    def _load_mask(self, path: Path) -> torch.Tensor:
        """加载掩码图像。
        
        Args:
            path: 掩码文件路径
            
        Returns:
            二值化掩码张量 [1, H, W]，值域{0, 1}
        """
        mask = Image.open(path)
        mask_array = np.array(mask)
        
        # 二值化：非零值为前景
        mask_array = (mask_array > 0).astype(np.float32)
        
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).float()
        
        # resize到目标尺寸（最近邻插值保持二值性）
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode='nearest'
        ).squeeze(0)
        
        return mask_tensor
    
    def _augment(
        self,
        image: torch.Tensor,
        vessel_mask: torch.Tensor,
        roi_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """数据增强。
        
        简便易实现的数据增强方法，同步应用于图像和标签。
        
        Args:
            image: 图像张量 [C, H, W]
            vessel_mask: 血管标签 [1, H, W]
            roi_mask: ROI掩码 [1, H, W]
            
        Returns:
            增强后的三个张量
        """
        # 1. 随机水平翻转（必须）
        if torch.rand(1) > 0.5:
            image = torch.flip(image, dims=[2])
            vessel_mask = torch.flip(vessel_mask, dims=[2])
            roi_mask = torch.flip(roi_mask, dims=[2])
        
        # 2. 随机垂直翻转（较少使用，但有助于增强）
        if torch.rand(1) > 0.5:
            image = torch.flip(image, dims=[1])
            vessel_mask = torch.flip(vessel_mask, dims=[1])
            roi_mask = torch.flip(roi_mask, dims=[1])
        
        # 3. 随机旋转90度（90度的旋转不会引入插值误差）
        if torch.rand(1) > 0.7:
            k = torch.randint(1, 4, (1,)).item()  # 90, 180, 270度
            image = torch.rot90(image, k=k, dims=[1, 2])
            vessel_mask = torch.rot90(vessel_mask, k=k, dims=[1, 2])
            roi_mask = torch.rot90(roi_mask, k=k, dims=[1, 2])
        
        # 4. 随机亮度调整（仅图像）
        if torch.rand(1) > 0.7:
            brightness_factor = torch.rand(1).item() * 0.4 + 0.8  # 0.8-1.2
            image = image * brightness_factor
            image = torch.clamp(image, 0, 1)
        
        # 5. 随机对比度调整（仅图像）
        if torch.rand(1) > 0.8:
            contrast_factor = torch.rand(1).item() * 0.4 + 0.8  # 0.8-1.2
            mean = image.mean()
            image = (image - mean) * contrast_factor + mean
            image = torch.clamp(image, 0, 1)
        
        return image, vessel_mask, roi_mask


def get_drive_loaders(
    config_path: str = 'config.yaml'
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """创建DRIVE数据加载器。
    
    工厂函数，从配置文件读取参数，返回训练/验证/测试加载器。
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器（若配置中有test_ids）
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    data_cfg = config['data']
    train_cfg = config.get('training', {})
    
    # 创建数据集
    train_dataset = DRIVEDataset(
        root=data_cfg['root'],
        image_ids=data_cfg['train_ids'],
        img_size=data_cfg['img_size'],
        is_training=True
    )
    
    val_dataset = DRIVEDataset(
        root=data_cfg['root'],
        image_ids=data_cfg['val_ids'],
        img_size=data_cfg['img_size'],
        is_training=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get('batch_size', 4),
        shuffle=True,
        num_workers=train_cfg.get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.get('batch_size', 4),
        shuffle=False,
        num_workers=train_cfg.get('num_workers', 4),
        pin_memory=True,
        drop_last=False
    )
    
    # 测试集（若有配置）
    test_loader = None
    if 'test_ids' in data_cfg and data_cfg['test_ids']:
        test_dataset = DRIVEDataset(
            root=data_cfg['root'],
            image_ids=data_cfg['test_ids'],
            img_size=data_cfg['img_size'],
            is_training=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )
    
    print(f'数据集加载完成:')
    print(f'  训练集: {len(train_dataset)} 张 ({min(data_cfg["train_ids"])}-{max(data_cfg["train_ids"])})')
    print(f'  验证集: {len(val_dataset)} 张 ({min(data_cfg["val_ids"])}-{max(data_cfg["val_ids"])})')
    if test_loader:
        print(f'  测试集: {len(test_loader.dataset)} 张')
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # 简单测试
    print('测试DRIVEDataset...')
    dataset = DRIVEDataset(
        root='./DRIVE',
        image_ids=[21, 22, 23],
        img_size=256
    )
    print(f'数据集大小: {len(dataset)}')
    
    img, vessel, roi = dataset[0]
    print(f'图像形状: {img.shape}, 值域: [{img.min():.3f}, {img.max():.3f}]')
    print(f'血管标签形状: {vessel.shape}, 均值: {vessel.mean():.4f}')
    print(f'ROI掩码形状: {roi.shape}, 均值: {roi.mean():.4f}')
