#!/usr/bin/env python3
"""验证 Baseline 和 Topo 脚本使用的 ROI 是否一致。"""

import yaml
import torch
from data_combined import get_combined_loaders

def verify_roi():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 强制使用相同配置
    config['training']['batch_size'] = 1  # 单样本便于查看
    
    train_loader, val_loader, _ = get_combined_loaders(config)
    
    print("=" * 60)
    print("ROI 验证报告")
    print("=" * 60)
    print(f"Config ROI mode: {config.get('data', {}).get('kaggle_roi', {}).get('mode', 'default')}")
    print(f"Val loader samples: {len(val_loader.dataset)}")
    print()
    
    # 取前3个样本查看
    for i, batch in enumerate(val_loader):
        if i >= 3:
            break
        
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            images, vessels, rois = batch
        else:
            images = batch['image']
            vessels = batch['vessel']
            rois = batch['roi']
        
        roi = rois[0, 0] if rois.dim() == 4 else rois[0]
        
        roi_mean = roi.mean().item()
        roi_unique = torch.unique(roi).tolist()
        
        print(f"Sample {i}: roi_mean={roi_mean:.4f}, roi_unique={roi_unique}")
    
    print()
    print("结论:")
    print(f"- ROI mode 实际使用: {config.get('data', {}).get('kaggle_roi', {}).get('mode', 'fov')}")
    print(f"- roi_mean≈0.6 表示 FOV 模式（椭圆ROI约占60%面积）")
    print(f"- roi_mean=1.0 表示 ones 模式（全图）")

if __name__ == '__main__':
    verify_roi()
