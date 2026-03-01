# -*- coding: utf-8 -*-
"""
文件名: data_combined.py
项目: retina_ph_seg
功能: 双模式数据加载（DRIVE纯数据集 / Kaggle联合数据集）- 修复版
作者: AI Assistant
创建日期: 2026-02-19
修复日期: 2026-02-22（修复Kaggle模式mask匹配、ROI shape、通道归一化）

关键修复（v1.1）:
  1. 鲁棒mask查找：支持多种命名格式和扩展名替换
  2. ROI shape统一：[1, H, W] 与 vessel一致
  3. 图像通道：保持RGB [3, H, W]，添加Normalize
  4. 找不到mask时raise ValueError（而非全黑mask）
"""

import os
import random
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
import yaml
from scipy import ndimage

# 尝试导入kagglehub
try:
    import kagglehub
    HAS_KAGGLEHUB = True
except ImportError:
    HAS_KAGGLEHUB = False

# 导入原有DRIVE数据加载器
from data_drive import get_drive_loaders as _get_drive_loaders_original


def get_combined_loaders(config: dict) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """双模式数据加载入口"""
    use_kaggle = config.get('data', {}).get('use_kaggle_combined', False)
    
    if not use_kaggle:
        print("=" * 60)
        print("纯DRIVE模式: 16 train + 4 val")
        print("=" * 60)
        return _get_drive_loaders_original(config_path='config.yaml')
    else:
        if not HAS_KAGGLEHUB:
            raise ValueError("需要kagglehub。安装: pip install kagglehub")
        return _get_kaggle_combined_loaders(config)


def _get_kaggle_combined_loaders(config: dict) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Kaggle联合数据集加载（修复版）"""
    
    # 优先检查本地路径
    local_path = Path("data/combined")
    train_images_path = local_path / "Training" / "images"
    
    if local_path.exists() and train_images_path.exists() and len(list(train_images_path.glob("*"))) > 0:
        dataset_path = local_path
        print("=" * 60)
        print("Kaggle联合模式: 使用本地数据集")
        print(f"路径: {dataset_path}")
        print("=" * 60)
    else:
        print("=" * 60)
        print("Kaggle联合模式: 自动下载数据集")
        print("=" * 60)
        dataset_path = kagglehub.dataset_download("pradosh123/retinal-vessel-segmentation-combined")
        dataset_path = Path(dataset_path)
    
    # 构建路径（支持大小写变体）
    train_dir = dataset_path / "Training"
    test_dir = dataset_path / "Test"
    
    train_img_dir = _find_dir(train_dir, ["images", "Images"])
    train_mask_dir = _find_dir(train_dir, ["masks", "Masks"])
    
    test_img_dir = _find_dir(test_dir, ["images", "Images"])
    test_mask_dir = _find_dir(test_dir, ["masks", "Masks"])
    
    # 统计
    train_images = _list_images(train_img_dir)
    test_images = _list_images(test_img_dir)
    
    print(f"训练集: {len(train_images)} 张")
    print(f"验证集: {len(test_images)} 张")
    print("=" * 60)
    
    data_cfg = config.get('data', {})
    roi_cfg = data_cfg.get('kaggle_roi', {})

    # 创建Dataset
    train_dataset = KaggleCombinedDataset(
        image_dir=train_img_dir,
        mask_dir=train_mask_dir,
        img_size=data_cfg['img_size'],
        augmentation=True,
        roi_mode=roi_cfg.get('mode', 'fov'),
        roi_threshold=roi_cfg.get('threshold', 8),
        roi_padding_scale=roi_cfg.get('padding_scale', 1.03)
    )
    
    val_dataset = KaggleCombinedDataset(
        image_dir=test_img_dir,
        mask_dir=test_mask_dir,
        img_size=data_cfg['img_size'],
        augmentation=False,
        roi_mode=roi_cfg.get('mode', 'fov'),
        roi_threshold=roi_cfg.get('threshold', 8),
        roi_padding_scale=roi_cfg.get('padding_scale', 1.03)
    )
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, None


def _find_dir(parent: Path, candidates: List[str]) -> Path:
    """查找存在的目录（支持大小写）"""
    for cand in candidates:
        p = parent / cand
        if p.exists():
            return p
    raise ValueError(f"在 {parent} 中未找到目录: {candidates}")


def _list_images(dir_path: Path) -> List[Path]:
    """列出所有图像文件"""
    exts = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.gif', '*.bmp', '*.PNG', '*.JPG', '*.TIF']
    files = []
    for ext in exts:
        files.extend(dir_path.glob(ext))
    return sorted(files)


class KaggleCombinedDataset(Dataset):
    """
    Kaggle联合数据集Dataset（修复版）
    
    关键修复:
    1. 鲁棒mask查找（支持多种命名格式）
    2. ROI shape [1, H, W]
    3. 图像RGB [3, H, W] + Normalize
    """
    
    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        img_size: int = 512,
        augmentation: bool = False,
        roi_mode: str = 'fov',
        roi_threshold: int = 8,
        roi_padding_scale: float = 1.03,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.augmentation = augmentation
        self.roi_mode = roi_mode.lower()
        self.roi_threshold = int(roi_threshold)
        self.roi_padding_scale = float(roi_padding_scale)
        
        self.image_files = _list_images(self.image_dir)
        if len(self.image_files) == 0:
            raise ValueError(f"未找到图像: {image_dir}")
        
        # 归一化（resnet34预训练需要）
        self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        print(f"  加载 {len(self.image_files)} 张图像")
        print(f"  ROI模式: {self.roi_mode} (threshold={self.roi_threshold}, padding_scale={self.roi_padding_scale})")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns: image [3,H,W], vessel_mask [1,H,W], roi_mask [1,H,W]"""
        img_path = self.image_files[idx]
        
        # 鲁棒mask查找（关键修复！）
        mask_path = self._find_mask(img_path)
        
        # 调试打印（仅在开发时开启）
        # print(f"Loading {img_path.name} → mask_found: {mask_path.exists()}")
        
        if not mask_path or not mask_path.exists():
            raise ValueError(f"Mask not found for {img_path}")
        
        # 加载图像（RGB）
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # 保持比例缩放 + Pad
        image, mask = self._resize_and_pad(image, mask)
        
        # 基于图像内容估计FOV ROI（在训练实际输入尺寸上计算）
        if self.roi_mode == 'ones':
            roi_mask = torch.ones((1, image.height, image.width), dtype=torch.float32)
        elif self.roi_mode == 'tiny':
            roi_mask = self._create_tiny_mask(image.height, image.width)
        else:
            roi_mask = self._create_fov_mask_from_image(image)

        # 转为tensor
        image = T.ToTensor()(image)  # [3, H, W]
        mask = T.ToTensor()(mask)    # [1, H, W]
        
        # 归一化（关键修复！）
        image = self.normalize(image)
        
        # 标签值域确保[0,1]（某些mask可能是0-255）
        if mask.max() > 1.0:
            mask = mask / 255.0
        
        # 数据增强
        if self.augmentation:
            image, mask, roi_mask = self._augment(image, mask, roi_mask)

        # 最终防御：保证ROI与输入尺寸完全一致
        if roi_mask.shape[-2:] != image.shape[-2:]:
            raise ValueError(f"ROI尺寸与图像不一致: roi={roi_mask.shape}, image={image.shape}")

        return image, mask, roi_mask

    def _create_fov_mask_from_image(self, image: Image.Image) -> torch.Tensor:
        """基于图像内容估计FOV，并拟合椭圆ROI。

        流程：阈值分离非黑区域 -> 最大连通域 -> 边界点主轴估计 -> 椭圆拟合。
        """
        img_np = np.array(image.convert('RGB'))
        gray = img_np.max(axis=2)  # 对彩色眼底边缘更稳健

        # 1) 阈值分离非黑区域
        binary = gray > self.roi_threshold
        if binary.sum() == 0:
            binary = gray > 0

        # 2) 最大连通域
        labeled, num = ndimage.label(binary)
        if num > 0:
            sizes = ndimage.sum(binary, labeled, index=np.arange(1, num + 1))
            largest_id = int(np.argmax(sizes)) + 1
            largest = labeled == largest_id
        else:
            largest = binary

        if largest.sum() < 32:
            # 极端异常时回退为全1，避免训练崩溃
            h, w = gray.shape
            return torch.ones((1, h, w), dtype=torch.float32)

        # 3) 边界点（用形态学边缘）
        eroded = ndimage.binary_erosion(largest, structure=np.ones((3, 3)), iterations=1)
        boundary = largest ^ eroded
        ys, xs = np.nonzero(boundary)
        if len(xs) < 16:
            ys, xs = np.nonzero(largest)

        # 4) 椭圆拟合：主轴旋转 + 包络半轴
        x = xs.astype(np.float32)
        y = ys.astype(np.float32)
        cx = float(x.mean())
        cy = float(y.mean())

        centered = np.stack([x - cx, y - cy], axis=0)
        cov = np.cov(centered)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]

        uv = eigvecs.T @ centered
        a = float(np.max(np.abs(uv[0])) * self.roi_padding_scale + 1.0)
        b = float(np.max(np.abs(uv[1])) * self.roi_padding_scale + 1.0)
        a = max(a, 4.0)
        b = max(b, 4.0)

        h, w = gray.shape
        yy, xx = np.mgrid[0:h, 0:w]
        grid = np.stack([xx - cx, yy - cy], axis=0).reshape(2, -1)
        uv_grid = eigvecs.T @ grid
        ellipse = ((uv_grid[0] / a) ** 2 + (uv_grid[1] / b) ** 2) <= 1.0
        ellipse = ellipse.reshape(h, w)

        # 限制在最大连通域附近，避免外扩到黑边
        roi = ellipse & ndimage.binary_dilation(largest, iterations=3)
        if roi.sum() < 64:
            roi = largest

        return torch.from_numpy(roi.astype(np.float32)).unsqueeze(0)
    
    def _create_tiny_mask(self, h: int, w: int, radius_ratio: float = 0.3) -> torch.Tensor:
        """创建极小的中心圆 ROI（用于极端测试）。
        
        Args:
            h: 图像高度
            w: 图像宽度
            radius_ratio: 半径占短边的比例（默认 0.3）
            
        Returns:
            roi_mask: [1, h, w] 中心圆 mask
        """
        import torch
        y = torch.arange(h).float()
        x = torch.arange(w).float()
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        center_y, center_x = h // 2, w // 2
        radius = min(h, w) * radius_ratio
        
        dist = torch.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
        mask = (dist < radius).float()
        
        return mask.unsqueeze(0)
    
    def _find_mask(self, img_path: Path) -> Optional[Path]:
        """
        鲁棒mask查找（支持多种Kaggle命名格式）
        """
        stem = img_path.stem
        ext_candidates = ['.png', '.tif', '.tiff', '.jpg', '.gif']
        
        # 尝试各种命名模式
        patterns = [
            # 直接同名
            f"{stem}",
            # _mask后缀
            f"{stem}_mask",
            f"{stem}_1stHO",
            f"{stem}_1st_manual",
            # mask_前缀
            f"mask_{stem}",
            # 去除可能的后缀再尝试
            stem.replace('_dr', '').replace('_g', '').replace('_h', ''),
        ]
        
        for pattern in patterns:
            for ext in ext_candidates:
                mask_path = self.mask_dir / (pattern + ext)
                if mask_path.exists():
                    return mask_path
        
        # 如果没找到，尝试模糊匹配（忽略大小写）
        for mask_file in self.mask_dir.iterdir():
            mask_stem_lower = mask_file.stem.lower()
            img_stem_lower = stem.lower()
            # 如果mask文件名包含图像stem的主要部分
            if img_stem_lower in mask_stem_lower or mask_stem_lower in img_stem_lower:
                return mask_file
        
        return None
    
    def _resize_and_pad(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """保持长宽比缩放 + Pad到目标尺寸"""
        w, h = image.size
        target = self.img_size
        
        # 计算缩放比例
        scale = min(target / w, target / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 缩放
        image = image.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)
        
        # Pad
        pad_left = (target - new_w) // 2
        pad_top = (target - new_h) // 2
        pad_right = target - new_w - pad_left
        pad_bottom = target - new_h - pad_top
        
        image = ImageOps.expand(image, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
        mask = ImageOps.expand(mask, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
        
        return image, mask
    
    def _augment(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        roi_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """数据增强（添加随机旋转-15°~+15°）"""
        # 水平翻转
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            roi_mask = TF.hflip(roi_mask)
        
        # 垂直翻转
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            roi_mask = TF.vflip(roi_mask)
        
        # 随机旋转 -15°~+15°（同时旋转image和mask）
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle, fill=0)
            # mask使用NEAREST插值保持二值性
            mask = TF.rotate(mask, angle, fill=0, interpolation=TF.InterpolationMode.NEAREST)
            roi_mask = TF.rotate(roi_mask, angle, fill=0, interpolation=TF.InterpolationMode.NEAREST)

        roi_mask = (roi_mask > 0.5).float()
        return image, mask, roi_mask


get_drive_loaders = get_combined_loaders


if __name__ == "__main__":
    print("运行验证协议...")
    
    config = yaml.safe_load(open("config.yaml"))
    config['data']['use_kaggle_combined'] = True
    
    print("\n测试Kaggle联合模式...")
    train, val, _ = get_combined_loaders(config)
    batch = next(iter(train))
    
    if isinstance(batch, (list, tuple)):
        img, vessel, roi = batch[0], batch[1], batch[2]
    else:
        img, vessel, roi = batch['image'][0], batch['vessel'][0], batch['roi'][0]
    
    print(f"\n验证结果:")
    print(f"  image.shape: {img.shape}")
    print(f"  vessel.shape: {vessel.shape}")
    print(f"  roi.shape: {roi.shape}")
    print(f"  roi.unique(): {torch.unique(roi)}")
    print(f"  vessel.max(): {vessel.max().item():.4f}")
    
    assert img.shape[0] == 3, f"必须RGB [3,H,W]， got {img.shape}"
    assert vessel.max() <= 1.0, f"标签未归一化: {vessel.max()}"
    assert roi.shape[0] == 1, f"ROI必须 [1,H,W]， got {roi.shape}"
    assert roi.sum() > 0, "ROI不应全0"
    assert roi.sum() < roi.numel(), "ROI不应全1（FOV模式）"
    
    print("\n✅ Kaggle模式修复成功！")
