#!/bin/bash
# Topo Loss ROI 限域证据对比实验
# 运行 ones vs tiny 两种 ROI 模式，各3 epoch，保存debug日志

set -e

echo "=============================================="
echo "Topo Loss ROI Debug 对比实验"
echo "=============================================="

# 创建输出目录
mkdir -p artifacts/roi_audit

# 备份原始 config
cp config.yaml config.yaml.bak

echo ""
echo "[Run A] ROI Mode: ones (全1 ROI，对照组)"
echo "----------------------------------------------"
# 修改 config: roi_mode=ones, debug_topo_roi=true, max_epochs=3
python -c "
import yaml
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
cfg['data']['use_kaggle_combined'] = True
cfg['data']['kaggle_roi'] = {'mode': 'ones'}
cfg['training']['debug_topo_roi'] = True
cfg['training']['max_epochs'] = 3
cfg['training']['enable_early_stopping'] = False
with open('config.yaml', 'w') as f:
    yaml.dump(cfg, f)
print('Config updated for ones ROI')
"

# 运行训练并保存日志
python train_with_topology.py 2>&1 | tee artifacts/roi_audit/topo_roi_debug_ones.txt | grep -E "(Epoch|TopoROI-Debug|Train Loss|Val Dice)"

echo ""
echo "[Run B] ROI Mode: tiny (极小中心圆ROI，测试组)"
echo "----------------------------------------------"
# 修改 config: roi_mode=tiny
python -c "
import yaml
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
cfg['data']['kaggle_roi'] = {'mode': 'tiny'}
with open('config.yaml', 'w') as f:
    yaml.dump(cfg, f)
print('Config updated for tiny ROI')
"

# 运行训练并保存日志
python train_with_topology.py 2>&1 | tee artifacts/roi_audit/topo_roi_debug_tiny.txt | grep -E "(Epoch|TopoROI-Debug|Train Loss|Val Dice)"

# 恢复原始 config
mv config.yaml.bak config.yaml

echo ""
echo "=============================================="
echo "实验完成！"
echo "日志文件:"
echo "  - artifacts/roi_audit/topo_roi_debug_ones.txt"
echo "  - artifacts/roi_audit/topo_roi_debug_tiny.txt"
echo "=============================================="
