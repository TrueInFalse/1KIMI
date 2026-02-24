# 运行命令手册

本手册存储项目所有bash运行指令，按使用场景分类。

---

## Stage 0: 数据验证

```bash
# 验证数据集（检查16+4划分、血管标签正确性）
python verify_data.py

# 输出：
# - debug_sample.png（可视化样本）
# - 控制台日志（划分验证、数值范围检查）
```

---

## Stage 1: U-Net基线训练

```bash
# 训练模型（完整200 epoch或早停）
python train_baseline.py

# 输出：
# - checkpoints/best_model.pth（最佳模型）
# - logs/training_log.csv（训练日志）
```

```bash
# 生成训练曲线图（训练结束后）
python visualize_results.py --plot-curves

# 输出：
# - results/training_curves.png（2×3布局，6个指标）
```

```bash
# 评估验证集
python evaluate.py --split val

# 评估测试集
python evaluate.py --split test

# 输出：
# - results/evaluation_results.png（可视化对比图）
# - results/predictions/*.npy（预测概率图）
```

```bash
# 可视化验证集和测试集样本
python visualize_results.py

# 指定样本索引
python visualize_results.py --val-idx 2 --test-idx 5

# 输出：
# - results/val_sample_comparison.png（验证集对比图）
# - results/test_sample_prediction.png（测试集预测图）
```

---

## Stage 2: 端到端拓扑正则训练

采用STE（Straight-Through Estimator）策略：
- **Forward**: CubicalRipser计算硬持续景观（CPU）
- **Backward**: 软Betti数提供梯度（GPU）
- **分辨率**: 512×512输出 → 128×128拓扑计算
- **λ预热**: 0-30ep=0.1, 30-100ep线性增至0.5

### 运行端到端训练

```bash
# 运行拓扑正则训练（需要先安装cripser）
python train_with_topology.py

# 输出：
# - checkpoints/best_model_topo.pth（最佳模型）
# - logs/training_topo_log.csv（训练日志）
```

### 拓扑损失参数（config.yaml）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `target_betti0` | 5.0 | 目标连通分量数 |
| `downsample` | 128 | 拓扑计算分辨率 |
| `lambda_start` | 0.1 | λ初始值（0-30ep） |
| `lambda_end` | 0.5 | λ最终值（100ep后） |
| `compute_freq` | 5 | 拓扑计算频率（每N iter） |

### 安装依赖

```bash
# 安装CubicalRipser（必需）
pip install -U cripser
```

---

## 辅助脚本

```bash
# 检查模型结构
python model_unet.py

# 检查数据加载
python data_drive.py

# 检查指标计算
python utils_metrics.py

# 检查拓扑损失
python topology_loss.py
```

---

## 批量运行（完整流程）

```bash
# 完整Stage 1流程
python verify_data.py && \
python train_baseline.py && \
python visualize_results.py --plot-curves && \
python evaluate.py --split val
```

---

## 常用调试命令

```bash
# 查看GPU使用
nvidia-smi

# 清理Python缓存
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# 查看训练日志
tail -f logs/training_log.csv

# 检查模型文件大小
ls -lh checkpoints/

# 检查输出文件
ls -lh results/
```

---

## 更新记录

| 日期 | 更新内容 |
|------|----------|
| 2026-02-07 | 初始版本，Stage 0-1命令 |
| 2026-02-09 | 移除TTA命令，准备端到端训练 |
