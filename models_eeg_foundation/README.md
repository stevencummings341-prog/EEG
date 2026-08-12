# EEG/ERP Foundation Models (DualCD Edition)

自包含的 EEG/ERP 基础模型包，包含完整的 **OrthogonalMask + DualPerturbation** (DualCD) 机制。
可直接用于运动想象、ERP 分类等脑电信号任务。

## 模型一览

```
UnifiedDINODualCD_Transformer (原始基线，含 DualCD)
│
├─ 改骨架: UnifiedDINODualCD_S4_Flatten    (Transformer → S4)
├─ 改池化: UnifiedDINODualCD_S4_Pos        (Transformer → S4, flatten → 注意力池化)
└─ 改池化: UnifiedDINODualCD_S4_Timepatch  (Transformer → S4, flatten → 时间分块池化)

独立: S4ERP (纯 S4 监督学习，无 DualCD)
```

| 模型 | 骨架 | 池化 | DualCD | 参数量 |
|:---|:---|:---|:---:|:---:|
| `S4ERP` | S4 | flatten | ❌ | ~900K |
| `UnifiedDINODualCD_S4_Pos` | S4 | attention | ✅ | ~2M |
| `UnifiedDINODualCD_S4_Timepatch` | S4 | temporal bin | ✅ | ~3.3M |
| `UnifiedDINODualCD_S4_Flatten` | S4 | flatten | ✅ | ~11M |
| `UnifiedDINODualCD_Transformer` | Transformer | flatten | ✅ | ~15M |

### DualCD 机制说明

所有 DualCD 模型包含：
- **OrthogonalMask**: 将特征分为 z_causal（因果/标签相关）和 z_spurious（虚假/标签无关）
- **DualPerturbation**: 类内扰动（替换 spurious）+ 类间扰动（替换 causal），迫使模型学习不变特征
- **PrototypeBank**: 每类 K 个原型，EMA 更新，用于熟悉度估计

## 快速使用

```python
from models_eeg_foundation import UnifiedDINODualCD_S4_Pos, S4ERP

# 配置
class Config:
    num_channels = 22
    num_classes = 4
    seq_len = 1000
    sampling_rate = 200.0

config = Config()
x = torch.randn(8, 1000, 22)
y = torch.randint(0, 4, (8,))

# 方式1: 监督学习 (最简单)
model = S4ERP(config)
out = model(x)
logits = out["logits"]

# 方式2: DualCD 训练 (推荐)
model = UnifiedDINODualCD_S4_Pos(config)
loss, parts = model.compute_loss(x, y, epoch=0)
# parts: {"dino", "ibot", "dkoleo", "base", "perturb", "proto"}
model.update_ema()
model.update_prototypes(x, y)

# 推理
logits = model(x)                # (8, 4)
features = model.encode(x)       # (8, 128) for Pos
```

## 模型选型建议

| 场景 | 推荐模型 | 理由 |
|:---|:---|:---|
| **快速验证** | `S4ERP` | 最简单，纯监督，无 DualCD |
| **小样本 (<5000)** | `UnifiedDINODualCD_S4_Pos` | DualCD + 注意力池化，参数最少 |
| **需要可解释性** | `UnifiedDINODualCD_S4_Timepatch` | 时间分块对应 ERP/MI 时间窗 |
| **大数据集** | `UnifiedDINODualCD_S4_Flatten` | 保留全部 patch 信息 |
| **与原论文对比** | `UnifiedDINODualCD_Transformer` | 原始 ERP Benchmark 架构 |

## 文件结构

```
models_eeg_foundation/
├── __init__.py          # 导出所有模型和组件
├── s4_layers.py         # S4 核心层 (HiPPO + FFT卷积)
├── pooling.py           # 3种池化策略
├── encoders.py          # ShallowNetEmbedding + S4/Transformer 编码器
├── losses.py            # DINO/iBOT/DKoleo/Prototype 损失
├── models.py            # 5个完整模型
├── README.md            # 本文件
└── comparison.md        # 架构对比与实验结果
```

## 依赖

- PyTorch >= 2.0
- NumPy
- 无其他外部依赖（S4 实现为纯 PyTorch）

## 训练建议

### DINO 模型的损失函数

```python
out = model(x)
loss = out["loss_total"]  # = loss_dino + loss_ibot + 0.1 * loss_dkoleo

# 各分量:
# loss_dino:   DINO 自蒸馏损失 (主要)
# loss_ibot:   掩码 patch 重建损失
# loss_dkoleo: 防坍缩损失 (权重 0.1)
```

### 学习率建议

| 模型 | 学习率 | Batch Size | Epochs |
|:---|:---:|:---:|:---:|
| S4ERP | 1e-3 | 128 | 100-300 |
| DINO 模型 | 1e-4 | 128 | 200-300 |

### 运动想象适配

```python
# 4 类运动想象 (left/right/foot/tongue)
config.num_classes = 4
config.num_channels = 22       # BCI Competition IV 2a
config.seq_len = 1000          # 5s @ 200Hz
config.sampling_rate = 200.0

# 时间分块池化可以自定义 bin 边界
# 例如 MI 的 ERD/ERS 通常在 8-30Hz, 0.5-4s
model = UnifiedDINO_S4_Timepatch(
    config,
    bin_boundaries_ms=[0, 500, 1000, 2000, 3000, 4000, 5000],
)
```
