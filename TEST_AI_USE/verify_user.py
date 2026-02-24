from data_drive import DRIVEDataset
ds = DRIVEDataset('/autodl-tmp/1KIMI/DRIVE')
img, mask = ds[0]
print(f"Image range: [{img.min():.3f}, {img.max():.3f}]")  # 应为[0,1]或[-1,1]
print(f"Mask unique: {torch.unique(mask)}")               # 应为[0., 1.]
print(f"Mask mean: {mask.mean():.4f}")                    # 应在0.06-0.12之间

from data_drive import get_drive_loaders
train_loader, val_loader = get_drive_loaders()
for imgs, masks in train_loader:
    print(f"Batch shape: {imgs.shape}")  # 应为 [4, 1, 256, 256]（batch=4, channel=1）
    print(f"Masks shape: {masks.shape}") # 应为 [4, 1, 256, 256]
    print(f"Mask sum: {masks.sum()}")    # 应为正数（有血管像素）
    break