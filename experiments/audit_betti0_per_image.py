# -*- coding: utf-8 -*-
"""Offline per-image Betti0 audit for the current ROI-aligned mainline."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".tmp_mpl"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
_mpl_lock = Path(os.environ["MPLCONFIGDIR"]) / "fontlist-v390.json.matplotlib-lock"
if _mpl_lock.exists():
    try:
        _mpl_lock.unlink()
    except OSError:
        pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy import ndimage
from tqdm import tqdm

from data_combined import get_combined_loaders
from model_unet import load_model
from utils_metrics import count_skeleton_fragments, skeletonize_vessel


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(config: Dict[str, Any]) -> str:
    requested = config.get("training", {}).get("device", "cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def normalize_min_sizes(min_sizes: Sequence[int]) -> List[int]:
    normalized = sorted({int(ms) for ms in min_sizes if int(ms) > 0})
    if not normalized:
        raise ValueError("min_sizes must contain at least one positive integer.")
    return normalized


def sanitize_label(label: Optional[str], fallback: str) -> str:
    raw = (label or fallback).strip()
    if not raw:
        raw = fallback
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
    return sanitized.strip("_") or fallback


def unpack_batch(batch: Any, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        images, vessels, rois = batch
    else:
        images = batch["image"]
        vessels = batch["vessel"]
        rois = batch["roi"]

    images = images.to(device)
    vessels = vessels.to(device)
    rois = rois.to(device)
    if rois.dim() == 3:
        rois = rois.unsqueeze(1)
    return images, vessels, rois


def dataset_has_labels(dataset: Any, split: str) -> bool:
    if split != "test":
        return True
    if getattr(dataset, "mode", None) == "test":
        return False
    return True


def sample_name(dataset: Any, global_index: int, batch_index: int, sample_index: int) -> str:
    image_ids = getattr(dataset, "image_ids", None)
    if image_ids is not None and global_index < len(image_ids):
        image_id = image_ids[global_index]
        if isinstance(image_id, (int, np.integer)):
            return f"{int(image_id):02d}"
        return str(image_id)

    image_files = getattr(dataset, "image_files", None)
    if image_files is not None and global_index < len(image_files):
        return Path(image_files[global_index]).stem

    return f"b{batch_index:03d}_s{sample_index:02d}"


def prepare_image_for_display(image: torch.Tensor) -> Tuple[np.ndarray, Dict[str, Any]]:
    image_np = image.detach().cpu().numpy()
    if image_np.ndim == 3 and image_np.shape[0] == 1:
        display = np.clip(image_np[0] * 0.5 + 0.5, 0.0, 1.0)
        return display, {"cmap": "gray"}

    if image_np.ndim == 3 and image_np.shape[0] >= 3:
        display = np.moveaxis(image_np[:3], 0, -1)
        display = np.clip(display * 0.5 + 0.5, 0.0, 1.0)
        return display, {}

    if image_np.ndim == 2:
        return np.clip(image_np, 0.0, 1.0), {"cmap": "gray"}

    if image_np.ndim == 3:
        return np.clip(image_np[0], 0.0, 1.0), {"cmap": "gray"}

    raise ValueError(f"Unsupported image shape for visualization: {image_np.shape}")


def label_components(binary: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
    labeled, num_features = ndimage.label(binary.astype(np.uint8))
    if num_features <= 0:
        return labeled, 0, np.empty(0, dtype=np.int64)
    sizes = np.bincount(labeled.ravel(), minlength=num_features + 1)[1 : num_features + 1]
    return labeled, int(num_features), sizes


def beta0_filtered_from_sizes(sizes: np.ndarray, min_size: int) -> int:
    if sizes.size == 0:
        return 0
    return int(np.count_nonzero(sizes >= int(min_size)))


def build_filtered_binary(labeled: np.ndarray, sizes: np.ndarray, min_size: int) -> np.ndarray:
    if sizes.size == 0:
        return np.zeros_like(labeled, dtype=np.uint8)
    lut = np.zeros(sizes.size + 1, dtype=np.uint8)
    lut[1:] = (sizes >= int(min_size)).astype(np.uint8)
    return lut[labeled].astype(np.uint8)


def colorize_components(labeled: np.ndarray, num_features: int) -> np.ndarray:
    if num_features <= 0:
        return np.zeros((*labeled.shape, 3), dtype=np.uint8)

    colors = np.zeros((num_features + 1, 3), dtype=np.uint8)
    for idx in range(1, num_features + 1):
        hue = (idx * 0.6180339887498949) % 1.0
        rgb = mcolors.hsv_to_rgb((hue, 0.75, 0.95))
        colors[idx] = np.round(rgb * 255).astype(np.uint8)

    return colors[labeled]


def audit_binary_components(
    binary: np.ndarray,
    min_sizes: Sequence[int],
    visual_min_size: int,
) -> Dict[str, Any]:
    labeled_raw, raw_beta0, sizes = label_components(binary)
    filtered_counts = {int(ms): beta0_filtered_from_sizes(sizes, int(ms)) for ms in min_sizes}

    filtered_binary_vis = build_filtered_binary(labeled_raw, sizes, visual_min_size)
    labeled_filtered, filtered_beta0_vis, _ = label_components(filtered_binary_vis)

    return {
        "raw_beta0": int(raw_beta0),
        "filtered_counts": filtered_counts,
        "labeled_raw": labeled_raw,
        "raw_num_features": int(raw_beta0),
        "labeled_filtered_vis": labeled_filtered,
        "filtered_num_features_vis": int(filtered_beta0_vis),
        "filtered_binary_vis": filtered_binary_vis,
    }


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def nanmean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def hash_array(array: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(array).tobytes()).hexdigest()


def visual_dir_name(run_label: str) -> str:
    return f"{run_label}_components_independent"


def gt_consistency_fields(min_sizes: Sequence[int]) -> List[str]:
    return [
        "gt_beta0_raw",
        "gt_skeleton_fragments",
        "roi_pixels",
        "gt_foreground_pixels",
    ] + [f"gt_beta0_filtered_ms{int(ms)}" for ms in min_sizes]


def pred_consistency_fields(min_sizes: Sequence[int]) -> List[str]:
    return [
        "pred_beta0_raw",
        "gt_beta0_raw",
        "delta_beta0_raw",
        "pred_skeleton_fragments",
        "gt_skeleton_fragments",
        "roi_pixels",
        "pred_foreground_pixels",
        "gt_foreground_pixels",
    ] + [f"pred_beta0_filtered_ms{int(ms)}" for ms in min_sizes] + [f"gt_beta0_filtered_ms{int(ms)}" for ms in min_sizes] + [
        f"delta_beta0_filtered_ms{int(ms)}" for ms in min_sizes
    ]


def assert_duplicate_rows_consistent(
    existing_row: Dict[str, Any],
    candidate_row: Dict[str, Any],
    fields: Sequence[str],
    *,
    base_image_id: str,
    row_kind: str,
) -> None:
    mismatches: List[str] = []
    for field in fields:
        if existing_row.get(field) != candidate_row.get(field):
            mismatches.append(f"{field}: existing={existing_row.get(field)!r}, duplicate={candidate_row.get(field)!r}")

    if mismatches:
        mismatch_preview = "; ".join(mismatches[:6])
        raise RuntimeError(
            "重复样本数值不一致，不能安全去重: "
            f"{row_kind} / base_image_id={base_image_id!r}. {mismatch_preview}"
        )


def assert_duplicate_visual_signature_consistent(
    existing_signature: Dict[str, str],
    candidate_signature: Dict[str, str],
    *,
    base_image_id: str,
) -> None:
    mismatches = [
        key
        for key in sorted(set(existing_signature.keys()) | set(candidate_signature.keys()))
        if existing_signature.get(key) != candidate_signature.get(key)
    ]
    if mismatches:
        raise RuntimeError(
            "重复样本数值不一致，不能安全去重: "
            f"visual payload / base_image_id={base_image_id!r}. mismatch_keys={mismatches}"
        )


def count_visualizations(visual_dir: Path) -> int:
    if not visual_dir.exists():
        return 0
    return sum(1 for _ in visual_dir.glob("*.png"))


def save_visualization(
    save_path: Path,
    image_display: np.ndarray,
    image_kwargs: Dict[str, Any],
    roi_binary: np.ndarray,
    gt_binary: np.ndarray,
    gt_raw_labeled: np.ndarray,
    gt_raw_num_features: int,
    gt_filtered_labeled: np.ndarray,
    gt_filtered_num_features: int,
    visual_min_size: int,
    pred_payload: Optional[Dict[str, Any]] = None,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if pred_payload is None:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
        axes = np.expand_dims(axes, axis=0)
    else:
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    axes[0, 0].imshow(image_display, **image_kwargs)
    axes[0, 0].set_title("Image")
    axes[0, 1].imshow(gt_binary, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("GT Mask (ROI)")
    axes[0, 2].imshow(colorize_components(gt_raw_labeled, gt_raw_num_features))
    axes[0, 2].set_title(f"GT Components Raw (beta0={gt_raw_num_features})")
    axes[0, 3].imshow(colorize_components(gt_filtered_labeled, gt_filtered_num_features))
    axes[0, 3].set_title(f"GT Components Filtered (ms={visual_min_size}, beta0={gt_filtered_num_features})")

    if pred_payload is not None:
        axes[1, 0].imshow(roi_binary, cmap="gray", vmin=0, vmax=1)
        axes[1, 0].set_title("ROI")
        axes[1, 1].imshow(pred_payload["pred_binary"], cmap="gray", vmin=0, vmax=1)
        axes[1, 1].set_title("Pred Mask (ROI)")
        axes[1, 2].imshow(
            colorize_components(
                pred_payload["pred_raw_labeled"],
                pred_payload["pred_raw_num_features"],
            )
        )
        axes[1, 2].set_title(f"Pred Components Raw (beta0={pred_payload['pred_raw_num_features']})")
        axes[1, 3].imshow(
            colorize_components(
                pred_payload["pred_filtered_labeled"],
                pred_payload["pred_filtered_num_features"],
            )
        )
        axes[1, 3].set_title(
            f"Pred Components Filtered (ms={visual_min_size}, beta0={pred_payload['pred_filtered_num_features']})"
        )

    for row_axes in axes:
        for ax in row_axes:
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def update_audit_report(
    report_path: Path,
    audit_csv_dir: Path,
    visual_root_dir: Path,
    split: str,
    min_sizes: Sequence[int],
    sample_summary: Dict[str, Any],
    focus_min_size: int = 20,
) -> None:
    gt_csv = audit_csv_dir / f"betti0_gt_{split}_per_image.csv"
    if not gt_csv.exists():
        return

    gt_rows = read_csv_rows(gt_csv)
    if not gt_rows:
        return

    focus_min_size = int(focus_min_size)
    independent_samples = int(sample_summary.get("unique_samples", len(gt_rows)))
    raw_entries = int(sample_summary.get("raw_entries", independent_samples))
    duplicate_entries = int(sample_summary.get("duplicate_entries", max(0, raw_entries - independent_samples)))
    duplicated_sample_keys = int(sample_summary.get("duplicated_sample_keys", 0))
    duplicate_detected = duplicate_entries > 0
    dedupe_key = str(sample_summary.get("dedupe_key", "base_image_id"))

    gt_raw_mean = nanmean(to_float(row.get("gt_beta0_raw")) for row in gt_rows)
    gt_ms_means = {
        int(ms): nanmean(to_float(row.get(f"gt_beta0_filtered_ms{int(ms)}")) for row in gt_rows)
        for ms in min_sizes
    }

    top_key = f"gt_beta0_filtered_ms{focus_min_size}"
    top_rows = sorted(
        gt_rows,
        key=lambda row: to_float(row.get(top_key)),
        reverse=True,
    )[:5]

    model_csv_paths = sorted(
        p
        for p in audit_csv_dir.glob(f"betti0_*_{split}_per_image.csv")
        if p.name != gt_csv.name
    )

    model_stats: Dict[str, Dict[str, float]] = {}
    output_count_lines: List[str] = []
    gt_visual_count = count_visualizations(visual_root_dir / visual_dir_name("gt"))
    output_count_lines.append(
        f"- `gt`: CSV rows = **{len(gt_rows)}**, visualizations = **{gt_visual_count}**"
    )

    for csv_path in model_csv_paths:
        match = re.match(rf"^betti0_(.+)_{re.escape(split)}_per_image\.csv$", csv_path.name)
        if not match:
            continue
        label = match.group(1)
        rows = read_csv_rows(csv_path)
        if not rows:
            continue
        pred_col = f"pred_beta0_filtered_ms{focus_min_size}"
        delta_col = f"delta_beta0_filtered_ms{focus_min_size}"
        model_stats[label] = {
            "row_count": float(len(rows)),
            "visual_count": float(count_visualizations(visual_root_dir / visual_dir_name(label))),
            "pred_mean_ms20": nanmean(to_float(row.get(pred_col)) for row in rows),
            "delta_mean_ms20": nanmean(to_float(row.get(delta_col)) for row in rows),
            "delta_mean_ms5": nanmean(to_float(row.get("delta_beta0_filtered_ms5")) for row in rows),
            "delta_mean_ms10": nanmean(to_float(row.get("delta_beta0_filtered_ms10")) for row in rows),
            "delta_mean_ms50": nanmean(to_float(row.get("delta_beta0_filtered_ms50")) for row in rows),
        }
        output_count_lines.append(
            f"- `{label}`: CSV rows = **{len(rows)}**, visualizations = **{int(model_stats[label]['visual_count'])}**"
        )

    baseline_key = next((k for k in model_stats.keys() if "baseline" in k.lower()), None)
    topo_key = next((k for k in model_stats.keys() if "topo" in k.lower()), None)

    gt_sensitivity = float("nan")
    if focus_min_size in gt_ms_means and np.isfinite(gt_ms_means[focus_min_size]):
        ref = max(gt_ms_means[focus_min_size], 1e-6)
        gt_sensitivity = (max(gt_ms_means.values()) - min(gt_ms_means.values())) / ref

    if np.isfinite(gt_ms_means.get(focus_min_size, float("nan"))) and gt_ms_means[focus_min_size] >= 15:
        gt_fragment_sentence = "GT itself is fragmented inside ROI, with a high number of connected components."
    elif np.isfinite(gt_ms_means.get(focus_min_size, float("nan"))) and gt_ms_means[focus_min_size] <= 6:
        gt_fragment_sentence = "GT is not heavily fragmented inside ROI, with relatively low component counts."
    else:
        gt_fragment_sentence = "GT shows moderate fragmentation inside ROI and should be checked per-image via visuals."

    if np.isfinite(gt_sensitivity) and gt_sensitivity >= 0.3:
        min_size_sentence = "Betti0/Delta-Beta0 are sensitive to min_size; threshold choice changes absolute values markedly."
    else:
        min_size_sentence = "Betti0/Delta-Beta0 are only mildly sensitive to min_size; conclusions are relatively stable."

    if baseline_key and topo_key:
        base_delta = model_stats[baseline_key]["delta_mean_ms20"]
        topo_delta = model_stats[topo_key]["delta_mean_ms20"]
        if np.isfinite(base_delta) and np.isfinite(topo_delta) and topo_delta < base_delta:
            topo_sentence = "At ms=20, topo candidate has lower Delta-Beta0 than baseline, suggesting better topology fit."
        elif np.isfinite(base_delta) and np.isfinite(topo_delta) and topo_delta > base_delta:
            topo_sentence = "At ms=20, topo candidate has higher Delta-Beta0 than baseline; topology error is not improved."
        else:
            topo_sentence = "Current baseline-vs-topo Delta-Beta0 comparison is inconclusive."
    else:
        topo_sentence = "Baseline/topo comparison is incomplete for now; rely on GT audit first."

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# BETTI0 AUDIT REPORT\n\n")
        f.write(f"- Split: `{split}`\n")
        f.write(f"- Independent samples: `{independent_samples}`\n")
        f.write(f"- Raw audit entries scanned: `{raw_entries}`\n")
        f.write(f"- Dedup key: `{dedupe_key}`\n\n")

        f.write("## Duplicate Handling\n\n")
        if duplicate_detected:
            f.write("- Duplicate samples detected: **yes**\n")
            f.write(f"- Duplicate raw entries removed: **{duplicate_entries}**\n")
            f.write(f"- Independent sample ids affected: **{duplicated_sample_keys}**\n")
            f.write(
                f"- Handling: validated duplicate rows by `{dedupe_key}`, kept the first occurrence, "
                "and dropped exact duplicate copies before writing CSV / report / visualizations.\n\n"
            )
        else:
            f.write("- Duplicate samples detected: **no**\n")
            f.write("- Handling: raw audit entries already matched independent samples; no deduplication was needed.\n\n")

        f.write("## Output Count Check\n\n")
        for line in output_count_lines:
            f.write(f"{line}\n")
        f.write("\n")

        f.write("## GT Statistics\n\n")
        f.write(f"- Mean `gt_beta0_raw`: **{gt_raw_mean:.4f}**\n")
        for ms in min_sizes:
            f.write(f"- Mean `gt_beta0_filtered_ms{ms}`: **{gt_ms_means[int(ms)]:.4f}**\n")
        f.write("\n")

        f.write(f"## Most Fragmented GT Samples (Top 5 by `gt_beta0_filtered_ms{focus_min_size}`)\n\n")
        for rank, row in enumerate(top_rows, start=1):
            value = to_float(row.get(top_key))
            f.write(f"{rank}. `{row.get('image_id', 'unknown')}`: {value:.4f}\n")
        f.write("\n")

        f.write("## Checkpoint Comparison (ms=20)\n\n")
        if not model_stats:
            f.write("- No checkpoint audit CSV found yet.\n\n")
        else:
            for label, stats in model_stats.items():
                f.write(
                    f"- `{label}`: mean `pred_beta0_filtered_ms20` = **{stats['pred_mean_ms20']:.4f}**, "
                    f"mean `delta_beta0_filtered_ms20` = **{stats['delta_mean_ms20']:.4f}**, "
                    f"CSV rows = **{int(stats['row_count'])}**, visuals = **{int(stats['visual_count'])}**\n"
                )
            f.write("\n")

        f.write("## One-line Interpretation\n\n")
        f.write(f"- {gt_fragment_sentence}\n")
        f.write(f"- {min_size_sentence}\n")
        f.write(f"- {topo_sentence}\n")
        f.write("- These findings affect topology metric interpretation in the thesis, and threshold/GT-fragmentation context should be stated.\n")


def run_audit(
    config_path: str,
    split: str,
    checkpoint: Optional[str],
    label: Optional[str],
    threshold: float,
    min_sizes: Sequence[int],
    max_samples: Optional[int],
    output_dir: Path,
    visual_root_dir: Path,
    report_path: Path,
) -> None:
    config = load_config(config_path)
    config = copy.deepcopy(config)
    device = resolve_device(config)
    min_sizes = normalize_min_sizes(min_sizes)
    visual_min_size = 20 if 20 in min_sizes else min_sizes[min(len(min_sizes) - 1, 0)]

    # Audit is an offline analysis task. Force single-process loading to avoid
    # multiprocessing pipe restrictions in constrained environments.
    config.setdefault("training", {})
    config["training"]["num_workers"] = 0

    train_loader, val_loader, test_loader = get_combined_loaders(config)
    loader_map = {"train": train_loader, "val": val_loader, "test": test_loader}
    dataloader = loader_map.get(split)
    if dataloader is None:
        raise ValueError(f'Split "{split}" is not available for current data mode.')

    has_labels = dataset_has_labels(dataloader.dataset, split)
    if not has_labels:
        raise ValueError("Selected split has no vessel labels. This audit requires GT labels.")

    model: Optional[torch.nn.Module] = None
    run_label = "gt"
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        model = load_model(checkpoint_path, device=device)
        run_label = sanitize_label(label, checkpoint_path.stem)
    else:
        run_label = "gt"

    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = visual_root_dir / visual_dir_name(run_label)
    visual_dir.mkdir(parents=True, exist_ok=True)

    gt_rows_by_key: Dict[str, Dict[str, Any]] = {}
    pred_rows_by_key: Dict[str, Dict[str, Any]] = {}
    visual_signatures_by_key: Dict[str, Dict[str, str]] = {}
    raw_counts_by_key: Dict[str, int] = {}
    raw_entries = 0

    processed = 0
    dataset = dataloader.dataset
    pbar = tqdm(dataloader, desc=f"Betti0 audit ({split}, label={run_label})")

    with torch.no_grad():
        for batch_index, batch in enumerate(pbar):
            images, vessels, rois = unpack_batch(batch, device)
            outputs = model(images) if model is not None else None

            for sample_index in range(images.shape[0]):
                if max_samples is not None and processed >= max_samples:
                    break

                base_image_id = sample_name(dataset, processed, batch_index, sample_index)
                image_id = base_image_id
                raw_counts_by_key[base_image_id] = raw_counts_by_key.get(base_image_id, 0) + 1
                raw_entries += 1
                image_display, image_kwargs = prepare_image_for_display(images[sample_index])

                roi_binary = (rois[sample_index, 0].detach().cpu().numpy() > 0).astype(np.uint8)
                gt_binary = (vessels[sample_index, 0].detach().cpu().numpy() > 0).astype(np.uint8)
                gt_binary = gt_binary * roi_binary

                gt_stats = audit_binary_components(gt_binary, min_sizes=min_sizes, visual_min_size=visual_min_size)
                gt_skeleton = skeletonize_vessel(gt_binary, roi_mask=None)
                gt_skeleton_fragments = count_skeleton_fragments(gt_skeleton, min_length=10)

                gt_row: Dict[str, Any] = {
                    "image_id": image_id,
                    "base_image_id": base_image_id,
                    "gt_beta0_raw": gt_stats["raw_beta0"],
                    "gt_skeleton_fragments": int(gt_skeleton_fragments),
                    "roi_pixels": int(np.count_nonzero(roi_binary)),
                    "gt_foreground_pixels": int(np.count_nonzero(gt_binary)),
                }
                for ms in min_sizes:
                    gt_row[f"gt_beta0_filtered_ms{ms}"] = gt_stats["filtered_counts"][int(ms)]

                pred_payload = None
                pred_row: Optional[Dict[str, Any]] = None
                if outputs is not None:
                    pred_prob = torch.sigmoid(outputs[sample_index, 0]).detach().cpu().numpy()
                    pred_binary = (pred_prob > threshold).astype(np.uint8)
                    pred_binary = pred_binary * roi_binary
                    pred_stats = audit_binary_components(
                        pred_binary,
                        min_sizes=min_sizes,
                        visual_min_size=visual_min_size,
                    )
                    pred_skeleton = skeletonize_vessel(pred_binary, roi_mask=None)
                    pred_skeleton_fragments = count_skeleton_fragments(pred_skeleton, min_length=10)

                    pred_row: Dict[str, Any] = {
                        "image_id": image_id,
                        "base_image_id": base_image_id,
                        "pred_beta0_raw": pred_stats["raw_beta0"],
                        "gt_beta0_raw": gt_stats["raw_beta0"],
                        "delta_beta0_raw": abs(pred_stats["raw_beta0"] - gt_stats["raw_beta0"]),
                        "pred_skeleton_fragments": int(pred_skeleton_fragments),
                        "gt_skeleton_fragments": int(gt_skeleton_fragments),
                        "roi_pixels": int(np.count_nonzero(roi_binary)),
                        "pred_foreground_pixels": int(np.count_nonzero(pred_binary)),
                        "gt_foreground_pixels": int(np.count_nonzero(gt_binary)),
                    }
                    for ms in min_sizes:
                        pred_col = f"pred_beta0_filtered_ms{ms}"
                        gt_col = f"gt_beta0_filtered_ms{ms}"
                        delta_col = f"delta_beta0_filtered_ms{ms}"
                        pred_row[pred_col] = pred_stats["filtered_counts"][int(ms)]
                        pred_row[gt_col] = gt_stats["filtered_counts"][int(ms)]
                        pred_row[delta_col] = abs(pred_row[pred_col] - pred_row[gt_col])

                    pred_payload = {
                        "pred_binary": pred_binary,
                        "pred_raw_labeled": pred_stats["labeled_raw"],
                        "pred_raw_num_features": pred_stats["raw_num_features"],
                        "pred_filtered_labeled": pred_stats["labeled_filtered_vis"],
                        "pred_filtered_num_features": pred_stats["filtered_num_features_vis"],
                    }

                visual_signature = {
                    "roi_binary_sha1": hash_array(roi_binary),
                    "gt_binary_sha1": hash_array(gt_binary),
                    "gt_raw_components_sha1": hash_array(gt_stats["labeled_raw"]),
                    "gt_filtered_components_sha1": hash_array(gt_stats["labeled_filtered_vis"]),
                }
                if pred_payload is not None:
                    visual_signature["pred_binary_sha1"] = hash_array(pred_payload["pred_binary"])
                    visual_signature["pred_raw_components_sha1"] = hash_array(pred_payload["pred_raw_labeled"])
                    visual_signature["pred_filtered_components_sha1"] = hash_array(pred_payload["pred_filtered_labeled"])

                if base_image_id in gt_rows_by_key:
                    assert_duplicate_rows_consistent(
                        gt_rows_by_key[base_image_id],
                        gt_row,
                        gt_consistency_fields(min_sizes),
                        base_image_id=base_image_id,
                        row_kind="gt",
                    )
                    assert_duplicate_visual_signature_consistent(
                        visual_signatures_by_key[base_image_id],
                        visual_signature,
                        base_image_id=base_image_id,
                    )
                    if pred_row is not None:
                        existing_pred = pred_rows_by_key.get(base_image_id)
                        if existing_pred is None:
                            raise RuntimeError(
                                "重复样本数值不一致，不能安全去重: "
                                f"pred row missing for duplicate base_image_id={base_image_id!r}"
                            )
                        assert_duplicate_rows_consistent(
                            existing_pred,
                            pred_row,
                            pred_consistency_fields(min_sizes),
                            base_image_id=base_image_id,
                            row_kind="pred",
                        )
                else:
                    gt_rows_by_key[base_image_id] = gt_row
                    if pred_row is not None:
                        pred_rows_by_key[base_image_id] = pred_row
                    visual_signatures_by_key[base_image_id] = visual_signature
                    save_visualization(
                        save_path=visual_dir / f"{image_id}.png",
                        image_display=image_display,
                        image_kwargs=image_kwargs,
                        roi_binary=roi_binary,
                        gt_binary=gt_binary,
                        gt_raw_labeled=gt_stats["labeled_raw"],
                        gt_raw_num_features=gt_stats["raw_num_features"],
                        gt_filtered_labeled=gt_stats["labeled_filtered_vis"],
                        gt_filtered_num_features=gt_stats["filtered_num_features_vis"],
                        visual_min_size=visual_min_size,
                        pred_payload=pred_payload,
                    )

                processed += 1

            if max_samples is not None and processed >= max_samples:
                break

    gt_rows = list(gt_rows_by_key.values())
    pred_rows = list(pred_rows_by_key.values())
    duplicate_entries = max(0, raw_entries - len(gt_rows))
    duplicated_sample_keys = sum(1 for count in raw_counts_by_key.values() if count > 1)
    sample_summary = {
        "dedupe_key": "base_image_id",
        "raw_entries": raw_entries,
        "unique_samples": len(gt_rows),
        "duplicate_entries": duplicate_entries,
        "duplicated_sample_keys": duplicated_sample_keys,
    }

    if duplicate_entries > 0:
        print(
            "[Audit][Warning] "
            f"Detected {duplicate_entries} duplicate raw entries across {duplicated_sample_keys} "
            f"independent sample ids on split={split}; using deduplicated rows keyed by base_image_id."
        )
    else:
        print(f"[Audit] No duplicate raw entries detected on split={split}; independent samples={len(gt_rows)}.")

    gt_fieldnames = [
        "image_id",
        "base_image_id",
        "gt_beta0_raw",
        "gt_skeleton_fragments",
        "roi_pixels",
        "gt_foreground_pixels",
    ] + [f"gt_beta0_filtered_ms{ms}" for ms in min_sizes]

    gt_csv = output_dir / f"betti0_gt_{split}_per_image.csv"
    write_csv(gt_csv, gt_fieldnames, gt_rows)

    if pred_rows:
        pred_fieldnames = [
            "image_id",
            "base_image_id",
            "pred_beta0_raw",
            "gt_beta0_raw",
            "delta_beta0_raw",
        ] + [f"pred_beta0_filtered_ms{ms}" for ms in min_sizes] + [f"gt_beta0_filtered_ms{ms}" for ms in min_sizes] + [
            f"delta_beta0_filtered_ms{ms}" for ms in min_sizes
        ] + [
            "pred_skeleton_fragments",
            "gt_skeleton_fragments",
            "roi_pixels",
            "pred_foreground_pixels",
            "gt_foreground_pixels",
        ]
        pred_csv = output_dir / f"betti0_{run_label}_{split}_per_image.csv"
        write_csv(pred_csv, pred_fieldnames, pred_rows)
        print(f"[Audit] Checkpoint CSV saved: {pred_csv}")

    update_audit_report(
        report_path=report_path,
        audit_csv_dir=output_dir,
        visual_root_dir=visual_root_dir,
        split=split,
        min_sizes=min_sizes,
        sample_summary=sample_summary,
        focus_min_size=20,
    )

    print(f"[Audit] GT CSV saved: {gt_csv}")
    print(f"[Audit] Visualization dir: {visual_dir}")
    print(f"[Audit] Report updated: {report_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Per-image Betti0 audit (ROI aligned, offline).")
    parser.add_argument("--config", type=str, default="config_125e.yaml", help="Path to config YAML.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path. Omit to run GT-only audit.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Label used in checkpoint CSV/visual directory names, e.g., baseline or topo_lcap020.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Prediction binarization threshold for checkpoint audit.",
    )
    parser.add_argument(
        "--min-sizes",
        type=int,
        nargs="+",
        default=[5, 10, 20, 50],
        help="Component size thresholds used for filtered Betti0.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit for debugging.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/audits",
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--visual-dir",
        type=str,
        default="results/betti0_audit_unique",
        help="Directory root for component visualization outputs.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="reports/BETTI0_AUDIT_REPORT.md",
        help="Markdown report path.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_audit(
        config_path=args.config,
        split=args.split,
        checkpoint=args.checkpoint,
        label=args.label,
        threshold=float(args.threshold),
        min_sizes=args.min_sizes,
        max_samples=args.max_samples,
        output_dir=Path(args.output_dir),
        visual_root_dir=Path(args.visual_dir),
        report_path=Path(args.report_path),
    )


if __name__ == "__main__":
    main()
