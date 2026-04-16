# BETTI0 AUDIT REPORT

- Split: `val`
- Independent samples: `20`
- Raw audit entries scanned: `40`
- Dedup key: `base_image_id`

## Duplicate Handling

- Duplicate samples detected: **yes**
- Duplicate raw entries removed: **20**
- Independent sample ids affected: **20**
- Handling: validated duplicate rows by `base_image_id`, kept the first occurrence, and dropped exact duplicate copies before writing CSV / report / visualizations.

## Output Count Check

- `gt`: CSV rows = **20**, visualizations = **20**
- `baseline`: CSV rows = **20**, visualizations = **20**
- `topo_lcap020`: CSV rows = **20**, visualizations = **20**

## GT Statistics

- Mean `gt_beta0_raw`: **985.2500**
- Mean `gt_beta0_filtered_ms5`: **149.7000**
- Mean `gt_beta0_filtered_ms10`: **57.0000**
- Mean `gt_beta0_filtered_ms20`: **19.0500**
- Mean `gt_beta0_filtered_ms50`: **6.1000**

## Most Fragmented GT Samples (Top 5 by `gt_beta0_filtered_ms20`)

1. `35_DRIVE`: 47.0000
2. `14_h_HRF`: 44.0000
3. `15_h_HRF`: 37.0000
4. `36_DRIVE`: 36.0000
5. `33_DRIVE`: 33.0000

## Checkpoint Comparison (ms=20)

- `baseline`: mean `pred_beta0_filtered_ms20` = **24.4500**, mean `delta_beta0_filtered_ms20` = **13.7000**, CSV rows = **20**, visuals = **20**
- `topo_lcap020`: mean `pred_beta0_filtered_ms20` = **3.6500**, mean `delta_beta0_filtered_ms20` = **15.6000**, CSV rows = **20**, visuals = **20**

## One-line Interpretation

- GT itself is fragmented inside ROI, with a high number of connected components.
- Betti0/Delta-Beta0 are sensitive to min_size; threshold choice changes absolute values markedly.
- At ms=20, topo candidate has higher Delta-Beta0 than baseline; topology error is not improved.
- These findings affect topology metric interpretation in the thesis, and threshold/GT-fragmentation context should be stated.
