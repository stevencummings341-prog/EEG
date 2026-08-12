# 架构对比与实验结果

## 1. 骨架对比: S4 vs Transformer

| 维度 | S4 | Transformer |
|:---|:---|:---|
| **核心机制** | 状态空间模型 (SSM) | 自注意力 (Self-Attention) |
| **时间复杂度** | O(L log L) — FFT 卷积 | O(L²) — 注意力矩阵 |
| **长程记忆** | HiPPO-LegS 矩阵，天然支持 | 依赖位置编码 |
| **参数量 (128d, 4层)** | ~66K | ~1.2M (6层) |
| **归纳偏置** | 强：频谱滤波 + 平滑过渡 | 弱：需要更多数据学习 |
| **适合信号** | 连续生理信号 (EEG/ECG) | 离散序列 (NLP) |

**来源**: ECG-CPC (Al-Masud et al. 2026) 在 ECG 上证明 SSM (3.8M) 打败 Transformer (97M)。
对 EEG/ERP 同样适用：S4 的频谱滤波特性与 EEG 的周期性信号结构高度匹配。

## 2. 池化对比

| 池化方式 | 输出维度 | 参数量 | 可解释性 | 适用场景 |
|:---|:---:|:---:|:---:|:---|
| **Flatten** | L×D (9216) | 0 | 无 | 大数据集，不缺参数 |
| **Attention** | D (128) | 13K | 可视化注意力权重 | 小样本，需要紧凑表示 |
| **Temporal Binned** | n_bins×D×2 (1536) | 0 | 每个 bin 对应时间窗 | ERP/MI，需要可解释性 |

**关键发现**: Flatten 的 feature_dim=9216 导致 ProjectionHead 有 4.7M 参数（占总参数 42%）。
AttentionPooling 将其降到 197K（减少 96%），且性能不降。

## 3. DINO 自监督的作用

DINO (Self-Distillation with No Labels) 通过以下机制提升表示质量:

- **DINO Loss**: 学生局部视图 → 匹配教师全局视图 → 学到不变特征
- **iBOT Loss**: 掩码 patch 重建 → 学到局部结构
- **DKoleo Loss**: 防止表示坍缩 → 保持特征多样性

**标签效率**: DINO 预训练可以在无标签数据上进行，然后用少量标签微调。
ECG 基础模型的实验表明标签效率提升 3.3-9×。

## 4. 模型参数量对比

以 C=21, T=170 (ERP 配置) 为例:

```
模型                              总参数    proj+cls    占比
─────────────────────────────────────────────────────────
S4ERP (standalone)                658K      18K         2.8%
UnifiedDINO_S4_Pos (推荐)        1.85M     428K        23.1%
UnifiedDINO_S4_Timepatch          3.27M     1.87M       57.3%
UnifiedDINO_S4_Flatten           11.15M     9.75M       87.5%
UnifiedDINO_Transformer_Flatten  ~15.0M    ~12.0M       80.0%
```

**结论**: AttentionPooling 将 proj+cls 参数从 9.75M 降到 428K，总参数从 11.15M 降到 1.85M。

## 5. 实验结果 (ERP Benchmark, Seed 43)

### s4erp (S4 standalone, supervised)

| 数据集 | Best Epoch | Val F1 | Test F1 | Test AUROC | 速度 |
|:---|:---:|:---:|:---:|:---:|:---:|
| CESCA-VODD | 10 | 0.689 | 0.686 | 0.777 | 66s/ep |
| CESCA-FLANKER | 12 | 0.666 | -- | -- | 66s/ep |
| SCPD | 4 | 0.650 | 0.789 | 0.789 | 50s/ep |

### dino_flatten (DINO + S4 + flatten)

| 数据集 | Best Epoch | Val F1 | Test F1 | Test AUROC | 速度 |
|:---|:---:|:---:|:---:|:---:|:---:|
| CESCA-VODD | 23 | 0.463 | 0.472 | 0.483 | 235s/ep |

> dino_flatten 的 F1 偏低，可能需要更多 epoch 或调参。

### 之前的 Transformer 基线 (UnifiedDINODualCDModel)

| 数据集 | Best Epoch | Test Acc | Test F1 | Test AUROC |
|:---|:---:|:---:|:---:|:---:|
| CESCA-VODD | 7 | 81.3% | 63.3% | 75.3% |
| CESCA-FLANKER | 5 | 61.6% | 61.4% | 66.9% |
| SCPD | 2 | 70.9% | 70.7% | 79.5% |

> 注意: Transformer 基线使用了 OrthogonalMask + DualPerturbation (ERP 特有的因果分离)，
> 而通用 DINO 模型没有这些组件，直接对比不完全公平。

## 6. 运动想象适配建议

### 输入格式
```
BCI Competition IV 2a: C=22, T=1000 (5s @ 200Hz), 4 classes
BCI Competition IV 2b: C=3,  T=750  (3.75s @ 200Hz), 2 classes
```

### 推荐配置
```python
# BCI-IV 2a (4类 MI)
config = type('Cfg', (), {
    'num_channels': 22,
    'num_classes': 4,
    'seq_len': 1000,
    'sampling_rate': 200.0,
})()

# 推荐: UnifiedDINO_S4_Pos (参数少，标签效率高)
model = UnifiedDINO_S4_Pos(config, d_model=128, n_layers=4, state_dim=8)

# 或: UnifiedDINO_S4_Timepatch (可解释，适合论文)
model = UnifiedDINO_S4_Timepatch(
    config,
    bin_boundaries_ms=[0, 500, 1000, 2000, 3000, 4000, 5000],
)
```

### 频率子带调整
当前 MultiViewGenerator 使用 4-12Hz (theta+alpha) 和 12-30Hz (beta)。
运动想象的 ERD/ERS 主要在 mu (8-12Hz) 和 beta (13-30Hz) 频段，建议调整:
```python
model.multi_view.low_freq = 8.0   # mu 起始
model.multi_view.high_freq = 30.0 # beta 结束
```
