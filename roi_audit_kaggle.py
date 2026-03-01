# -*- coding: utf-8 -*-
"""Kaggle ROI审计脚本。

产物：
A) 30 train + 10 val overlay
B) ROI面积占比统计与异常样本overlay
C) batch级别shape/unique打印
D) 同checkpoint对比 ones ROI vs fov ROI 指标差异
"""

from pathlib import Path
from typing import Dict, List
import random
import copy

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from data_combined import get_combined_loaders
from model_unet import load_model
from utils_metrics import compute_all_metrics, tensor_to_numpy


def _overlay(image: np.ndarray, roi: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    img = np.clip((image.transpose(1, 2, 0) * 0.5 + 0.5), 0, 1)
    ax.imshow(img)
    ax.imshow(roi, cmap='jet', alpha=0.28)
    ax.axis('off')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def _collect_overlays(loader, split: str, out_dir: Path, sample_n: int) -> List[float]:
    ds = loader.dataset
    n = min(sample_n, len(ds))
    idxs = random.sample(list(range(len(ds))), n)
    ratios = []
    for i, idx in enumerate(idxs):
        image, _, roi = ds[idx]
        roi_np = tensor_to_numpy(roi[0]) if roi.ndim == 3 else tensor_to_numpy(roi)
        ratio = float(roi_np.mean())
        ratios.append(ratio)
        _overlay(tensor_to_numpy(image), roi_np, out_dir / split / f'{split}_{idx:04d}_{i:02d}.png')
    return ratios


def _save_anomalies(loader, split: str, out_dir: Path, low=0.45, high=0.95) -> Dict[str, int]:
    ds = loader.dataset
    saved = 0
    total = len(ds)
    for idx in range(total):
        image, _, roi = ds[idx]
        roi_np = tensor_to_numpy(roi[0]) if roi.ndim == 3 else tensor_to_numpy(roi)
        ratio = float(roi_np.mean())
        if ratio < low or ratio > high:
            _overlay(tensor_to_numpy(image), roi_np, out_dir / 'anomaly' / split / f'{split}_{idx:04d}_ratio_{ratio:.3f}.png')
            saved += 1
    return {'total': total, 'saved': saved}


def _evaluate(model, loader, device: str) -> Dict[str, float]:
    model.eval()
    sums = {'dice': 0.0, 'cl_break': 0.0, 'delta_beta0': 0.0}
    n = 0
    with torch.no_grad():
        for images, vessels, rois in loader:
            images = images.to(device)
            outputs = model(images)
            for b in range(images.shape[0]):
                pred_np = torch.sigmoid(outputs[b, 0]).cpu().numpy()
                target_np = vessels[b, 0].cpu().numpy()
                roi_np = rois[b, 0].cpu().numpy() if rois.ndim == 4 else rois[b].cpu().numpy()
                m = compute_all_metrics(pred_np, target_np, roi_np, threshold=0.5, compute_topology=True)
                sums['dice'] += m.dice
                sums['cl_break'] += m.cl_break
                sums['delta_beta0'] += m.delta_beta0
                n += 1
    return {k: v / max(1, n) for k, v in sums.items()}


def main(config_path: str = 'config.yaml') -> None:
    random.seed(42)
    out_dir = Path('results/roi_audit')
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config_fov = copy.deepcopy(config)
    config_fov['data']['use_kaggle_combined'] = True
    config_fov['data'].setdefault('kaggle_roi', {})['mode'] = 'fov'

    train_loader, val_loader, _ = get_combined_loaders(config_fov)

    # C: batch级别检查
    images, _, rois = next(iter(train_loader))
    unique_vals = torch.unique(rois)
    print(f'[Batch检查] image.shape={tuple(images.shape)}, roi.shape={tuple(rois.shape)}, roi.unique={unique_vals.tolist()}')

    # A: 抽样overlay
    train_ratios = _collect_overlays(train_loader, 'train', out_dir, 30)
    val_ratios = _collect_overlays(val_loader, 'val', out_dir, 10)

    # B: 面积占比统计 + 异常样本
    all_ratios = train_ratios + val_ratios
    stats = {
        'min': float(np.min(all_ratios)),
        'median': float(np.median(all_ratios)),
        'max': float(np.max(all_ratios)),
    }
    print(f"[ROI面积占比] min={stats['min']:.4f}, median={stats['median']:.4f}, max={stats['max']:.4f}")

    an_train = _save_anomalies(train_loader, 'train', out_dir)
    an_val = _save_anomalies(val_loader, 'val', out_dir)
    print(f"[异常样本] train={an_train['saved']}/{an_train['total']}, val={an_val['saved']}/{an_val['total']}")

    # D: 同checkpoint对比 ones vs fov
    ckpt_candidates = [Path('checkpoints/best_model_topo.pth'), Path('checkpoints/best_model.pth')]
    ckpt = next((p for p in ckpt_candidates if p.exists()), None)
    if ckpt is None:
        print('[评估对比] 跳过：未找到checkpoint')
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(ckpt, device=device)

    fov_metrics = _evaluate(model, val_loader, device)

    config_ones = copy.deepcopy(config_fov)
    config_ones['data']['kaggle_roi']['mode'] = 'ones'
    _, val_loader_ones, _ = get_combined_loaders(config_ones)
    ones_metrics = _evaluate(model, val_loader_ones, device)

    print('[评估对比] same checkpoint')
    print(f'  checkpoint: {ckpt}')
    print(f"  FOV ROI  : Dice={fov_metrics['dice']:.4f}, CL-Break={fov_metrics['cl_break']:.2f}, Δβ₀={fov_metrics['delta_beta0']:.2f}")
    print(f"  Ones ROI : Dice={ones_metrics['dice']:.4f}, CL-Break={ones_metrics['cl_break']:.2f}, Δβ₀={ones_metrics['delta_beta0']:.2f}")
    print(f"  Δ(FOV-Ones): Dice={fov_metrics['dice']-ones_metrics['dice']:+.4f}, CL-Break={fov_metrics['cl_break']-ones_metrics['cl_break']:+.2f}, Δβ₀={fov_metrics['delta_beta0']-ones_metrics['delta_beta0']:+.2f}")


if __name__ == '__main__':
    main()
