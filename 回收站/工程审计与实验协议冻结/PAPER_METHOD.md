# 论文方法段落素材 (Paper Method Section)

**字数**: 约500字  
**适用位置**: Methods → Training Strategy / Topological Regularization

---

## 方法段落（可直接使用）

### 段落1: 问题与动机

视网膜血管分割面临血管断裂挑战，传统像素级损失（Dice）难以建模拓扑连通性。我们引入持续同调（Persistent Homology, PH）作为拓扑正则，强制网络学习血管的整体连通结构。

### 段落2: Filtration定义与PH计算

对于分割概率图 $P \in [0,1]^{H \times W}$，我们构建子水平集filtration：
$$f(x) = 1 - P(x)$$
其中高概率区域对应低filtration值（早birth）。使用cripser库计算0维持续同调，得到各连通分量的birth-death对 $(b_i, d_i)$，持续时间为 $\tau_i = d_i - b_i$。

### 段落3: 拓扑损失定义

我们的拓扑损失鼓励预测结果具有与真实标注相似的连通分量数：

1. **保留主分量**: 仅保留前 $K=5$ 个最长持续的连通分量（对应主要血管树）
2. **MSE约束**: 强制这些分量的持续时间接近目标值0.5：
   $$\mathcal{L}_{\text{topo}} = \frac{1}{K} \sum_{i=1}^{K} (\tau_i - 0.5)^2 \times \alpha$$
   其中 $\alpha=100$ 为损失缩放因子，使拓扑损失与Dice损失量级匹配。

### 段落4: 总损失与λ调度

总损失为Dice损失与拓扑损失的加权和：
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Dice}} + \lambda(t) \cdot \mathcal{L}_{\text{topo}}$$

采用**015课程学习策略**：
- **阶段1** (前30% epochs): $\lambda=0$，纯Dice预热
- **阶段2** (中间30% epochs): $\lambda: 0 \rightarrow 0.1$，轻度拓扑约束
- **阶段3** (最后40% epochs): $\lambda: 0.1 \rightarrow 0.5$，加强拓扑优化

该策略避免早期拓扑约束过强干扰基础特征学习。

### 段落5: 评估指标与阈值策略

采用像素精度与拓扑连通性双指标评估：
- **Dice系数**: 像素级重叠度量
- **CL-Break**: $|C_{\text{pred}} - C_{\text{gt}}|$，连通分量数差异
- **$\Delta\beta_0$**: $C_{\text{pred}} - C_{\text{gt}}$，带符号偏差

二值化阈值通过验证集扫描确定（范围0.3-0.7，步长0.05），最终**冻结阈值为0.5**，确保测试集评估的公平性。

### 段落6: 副作用监控

为防止拓扑约束导致的过度连通（over-connecting），我们监控假阳性率（FPR）变化。实验表明在$\lambda_{\text{end}}=0.5$时，FPR仅上升22%，细血管粘连案例可控。

---

## 公式速查

| 符号 | 含义 |
|------|------|
| $P$ | 分割概率图 [H,W] |
| $f = 1-P$ | sublevel set filtration |
| $(b_i, d_i)$ | birth-death对 |
| $\tau_i = d_i - b_i$ | 第i个连通分量持续时间 |
| $K=5$ | 保留的最大分量数 (target_beta0) |
| $\alpha=100$ | 损失缩放因子 (loss_scale) |
| $\lambda(t)$ | 时变权重 (015策略) |

---

## 图表建议

### 图1: 015策略示意图
```
λ
0.5 |                    _______
0.4 |               ____/
0.3 |          ____/
0.2 |     ____/
0.1 |____/
  0 |____|____|____|____|
    0    60   120  200   epochs
    [P1] [P2] [P3]
```

### 图2: 拓扑损失作用机制
```
[断裂预测]      [PH计算]       [梯度回传]
  ━   ━  ━  ->  多短barcode  ->  惩罚短持续
   ↓    ↓
  ═══════      [修复后]       [目标]
  ━━━━━━━━━  <-  长barcode   <-  鼓励长持续
```

---

## 与代码对应

| 论文描述 | 代码位置 |
|---------|---------|
| $f = 1-P$ | `topology_loss.py: filtration = 1.0 - prob` |
| 0维PH计算 | `cripser.compute_ph_torch(..., maxdim=0)` |
| 保留前K个 | `torch.topk(lifetimes, target_beta0)` |
| MSE到0.5 | `F.mse_loss(lifetimes, 0.5) * loss_scale` |
| 015策略 | `LambdaScheduler` in `train_with_topology.py` |
| 阈值0.5 | `config.yaml: topology_threshold: 0.5` |

---

**版本**: v1.0  
**更新**: 2026-02-25
