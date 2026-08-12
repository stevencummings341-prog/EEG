# models_eeg_foundation 使用指南

> 从零开始，把 EEG/ERP 基础模型用到你的数据集上。

---

## 0. 前置条件

```
Python >= 3.9
PyTorch >= 2.0
NumPy
```

没有其他依赖。S4 实现是纯 PyTorch，不需要装 `s4` 或 `mamba` 包。

---

## 1. 复制包到你的项目

```bash
# 把整个文件夹复制到你的项目目录
cp -r models_eeg_foundation/ /your/project/models_eeg_foundation/
```

目录结构：
```
your_project/
├── models_eeg_foundation/    ← 复制过来的
│   ├── __init__.py
│   ├── s4_layers.py
│   ├── pooling.py
│   ├── encoders.py
│   ├── losses.py
│   ├── models.py
│   └── USAGE_GUIDE.md
├── train.py                  ← 你自己的训练脚本
└── data/                     ← 你的数据
```

---

## 2. 准备数据

模型接受的输入格式：

```python
x = torch.Tensor  # (batch, time_steps, channels)
y = torch.Tensor  # (batch,) 整数类别标签
```

### 常见数据集示例

**BCI Competition IV 2a（运动想象 4 类）**：
```python
# 原始: (trials, channels, time) = (288, 22, 1000)
# 转换: (trials, time, channels) = (288, 1000, 22)
x = raw_data.transpose(0, 2, 1)  # (N, T, C)
y = labels  # 0=left, 1=right, 2=foot, 3=tongue

config.num_channels = 22
config.num_classes = 4
config.seq_len = 1000       # 5秒 @ 200Hz
config.sampling_rate = 200.0
```

**BCI Competition IV 2b（运动想象 2 类）**：
```python
x = raw_data.transpose(0, 2, 1)  # (N, 750, 3)
y = labels  # 0=left, 1=right

config.num_channels = 3
config.num_classes = 2
config.seq_len = 750        # 3.75秒 @ 200Hz
config.sampling_rate = 200.0
```

**P300 Speller**：
```python
# 典型: (trials, channels, time) = (N, 64, 200)
x = raw_data.transpose(0, 2, 1)  # (N, 200, 64)
y = labels  # 0=non-target, 1=target

config.num_channels = 64
config.num_classes = 2
config.seq_len = 200        # 1秒 @ 200Hz
config.sampling_rate = 200.0
```

**Sleep Staging（多分类）**：
```python
# 典型: (trials, channels, time) = (N, 2, 3000)
x = raw_data.transpose(0, 2, 1)  # (N, 3000, 2)
y = labels  # 0=W, 1=N1, 2=N2, 3=N3, 4=REM

config.num_channels = 2
config.num_classes = 5
config.seq_len = 3000       # 30秒 @ 100Hz
config.sampling_rate = 100.0
```

### 数据预处理建议

```python
# 1. 逐样本 z-score 归一化（推荐）
def normalize(x):
    """x: (N, T, C)"""
    m = x.mean(axis=1, keepdims=True)
    s = x.std(axis=1, keepdims=True).clip(min=1e-8)
    return (x - m) / s

x = normalize(x)

# 2. 或者逐通道归一化
def normalize_per_channel(x):
    """x: (N, T, C)"""
    m = x.mean(axis=(0, 1), keepdims=True)
    s = x.std(axis=(0, 1), keepdims=True).clip(min=1e-8)
    return (x - m) / s
```

---

## 3. 选择模型

### 决策树

```
你的数据集有多少样本？
│
├─ < 2000（小样本）
│   └─ 用 UnifiedDINODualCD_S4_Pos（DualCD + 注意力池化，参数最少）
│
├─ 2000 ~ 10000（中等）
│   ├─ 需要可解释性？→ UnifiedDINODualCD_S4_Timepatch
│   └─ 不需要？→ UnifiedDINODualCD_S4_Flatten
│
└─ > 10000（大样本）
    └─ UnifiedDINODualCD_S4_Flatten 或 UnifiedDINODualCD_Transformer
```

### 快速验证

先用 `S4ERP`（最简单，纯监督）跑通流程，再换 DualCD 模型。

```python
from models_eeg_foundation import S4ERP
model = S4ERP(config)  # 最快验证
```

---

## 4. 训练代码

### 4.1 纯监督训练（S4ERP）

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from models_eeg_foundation import S4ERP

# ── 配置 ──
class Config:
    num_channels = 22
    num_classes = 4
    seq_len = 1000
    sampling_rate = 200.0

config = Config()

# ── 数据 ──
# x_train: (N, T, C), y_train: (N,)
train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(x_train), torch.LongTensor(y_train)),
    batch_size=64, shuffle=True,
)

# ── 模型 ──
model = S4ERP(config, d_model=128, n_layers=4, state_dim=8)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# ── 训练 ──
for epoch in range(100):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        out = model(x_batch)
        loss = torch.nn.functional.cross_entropy(out["logits"], y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    scheduler.step()
    print(f"Epoch {epoch}: loss={total_loss/len(y_train):.4f}")
```

### 4.2 DualCD 训练（推荐）

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from models_eeg_foundation import UnifiedDINODualCD_S4_Pos

config = Config()
model = UnifiedDINODualCD_S4_Pos(
    config,
    d_model=128,
    n_layers=4,
    state_dim=8,
    lambda_intra=0.5,      # 类内扰动权重
    dino_out_dim=256,       # DINO 投影维度
    proto_k=5,              # 每类原型数
    teacher_momentum=0.996, # EMA 动量
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(x_train), torch.LongTensor(y_train)),
    batch_size=128, shuffle=True,
)

for epoch in range(300):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()

        # DualCD 训练：DINO + iBOT + DKoleo + 分类 + 扰动 + 原型
        loss, parts = model.compute_loss(x_batch, y_batch, epoch=epoch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
        optimizer.step()

        # 更新 Teacher EMA 和原型
        model.update_ema()
        model.update_prototypes(x_batch, y_batch)

        total_loss += loss.item() * len(y_batch)

    scheduler.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss={total_loss/len(y_train):.4f} "
              f"dino={parts['dino'].item():.4f} "
              f"base={parts['base'].item():.4f} "
              f"perturb={parts['perturb'].item():.4f}")
```

### 4.3 损失函数说明

```python
loss, parts = model.compute_loss(x, y, epoch=0)

# parts 包含:
# parts["dino"]    - DINO 自蒸馏损失（学生局部 vs 教师全局）
# parts["ibot"]    - 掩码 patch 重建损失
# parts["dkoleo"]  - 防坍缩损失（权重 0.1）
# parts["base"]    - 分类交叉熵损失（在 z_causal 上）
# parts["perturb"] - 双重扰动损失（类内 + 类间）
# parts["proto"]   - 原型熟悉度损失（权重 0.1）

# 总损失 = dino + ibot + 0.1*dkoleo + 0.5*base + 0.5*perturb + 0.1*proto
```

---

## 5. 推理与特征提取

```python
# ── 分类推理 ──
model.eval()
with torch.no_grad():
    logits = model(x_test)           # (N, num_classes)
    preds = logits.argmax(dim=1)     # (N,)
    probs = torch.softmax(logits, 1) # (N, num_classes)

# ── 特征提取（用于下游任务/可视化） ──
with torch.no_grad():
    features = model.encode(x_test)  # (N, feature_dim)
    # Pos:      (N, 128)
    # Timepatch: (N, 1536)
    # Flatten:  (N, patch_num * d_model)
```

---

## 6. 超参数建议

### S4ERP（纯监督）

| 参数 | 推荐值 | 说明 |
|:---|:---:|:---|
| d_model | 128 | 特征维度 |
| n_layers | 4 | S4 层数（比 Transformer 的 6 层少） |
| state_dim | 8 | S4 状态维度（8-16） |
| d_ff | 256 | FFN 中间维度 |
| lr | 1e-3 | 学习率 |
| batch_size | 128 | 批大小 |
| epochs | 100-300 | 训练轮数 |

### DualCD 模型

| 参数 | 推荐值 | 说明 |
|:---|:---:|:---|
| d_model | 128 | 特征维度 |
| n_layers | 4 | S4 层数 |
| state_dim | 8 | S4 状态维度 |
| lambda_intra | 0.5 | 类内扰动权重（0.3-0.7） |
| dino_out_dim | 256 | DINO 投影维度 |
| proto_k | 5 | 每类原型数 |
| teacher_momentum | 0.996 | EMA 动量 |
| lr | 1e-4 | 学习率（比纯监督低 10 倍） |
| batch_size | 128 | 批大小 |
| epochs | 200-300 | 训练轮数 |
| warmup_epochs | 5 | 扰动损失预热轮数 |

---

## 7. 运动想象适配细节

### 7.1 频率子带调整

MultiViewGenerator 默认用 4-12Hz 和 12-30Hz。运动想象的 ERD/ERS 主要在 mu (8-12Hz) 和 beta (13-30Hz)：

```python
# 在模型创建后修改
model.multi_view.low_freq = 8.0
model.multi_view.high_freq = 30.0
# 同时修改 teacher 的 multi_view
if hasattr(model, 'teacher_embedding'):
    pass  # multi_view 是共享的，不需要单独改
```

### 7.2 时间分块池化（适配 MI 时间窗）

```python
from models_eeg_foundation import UnifiedDINODualCD_S4_Timepatch

# MI 典型时间窗: 0.5-4s（cue 后）
model = UnifiedDINODualCD_S4_Timepatch(
    config,
    bin_boundaries_ms=[0, 500, 1000, 2000, 3000, 4000, 5000],
    use_std=True,
)
# 输出: 6 bins × 128 d_model × 2 (mean+std) = 1536 维
```

### 7.3 数据增强（可选）

```python
# 时间偏移
def time_shift(x, max_shift=50):
    shift = torch.randint(-max_shift, max_shift+1, (1,)).item()
    return torch.roll(x, shift, dims=1)

# 高斯噪声
def add_noise(x, std=0.1):
    return x + torch.randn_like(x) * std

# 通道 dropout
def channel_dropout(x, p=0.1):
    mask = torch.bernoulli(torch.ones(x.shape[2]) * (1 - p))
    return x * mask.unsqueeze(0).unsqueeze(0)
```

---

## 8. 评估代码

```python
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np

model.eval()
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        logits = model(x_batch)
        probs = torch.softmax(logits, 1)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(y_batch.numpy())
        all_probs.append(probs.cpu().numpy())

preds = np.concatenate(all_preds)
labels = np.concatenate(all_labels)
probs = np.concatenate(all_probs)

acc = accuracy_score(labels, preds)
f1 = f1_score(labels, preds, average='macro')
auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
cm = confusion_matrix(labels, preds)

print(f"Accuracy: {acc:.4f}")
print(f"Macro F1: {f1:.4f}")
print(f"Macro AUROC: {auc:.4f}")
print(f"Confusion Matrix:\n{cm}")
```

---

## 9. 常见问题

### Q: 训练很慢怎么办？
- S4 比 Transformer 快（O(L log L) vs O(L²)），但 DualCD 的多视图前向传播会增加开销
- 先用 `S4ERP` 验证流程，再换 DualCD 模型
- 减小 `d_model`（128→64）或 `n_layers`（4→2）

### Q: 显存不够怎么办？
- 减小 `batch_size`（128→64→32）
- 减小 `d_model`（128→64）
- 用 `UnifiedDINODualCD_S4_Pos`（feature_dim=128，最小）

### Q: 过拟合怎么办？
- 增加数据增强（时间偏移、高斯噪声、通道 dropout）
- 减小 `d_model` 或 `n_layers`
- 增加 `dropout`（0.1→0.2→0.3）
- DualCD 的 DualPerturbation 本身有正则化效果

### Q: 如何保存和加载模型？
```python
# 保存
torch.save({
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'epoch': epoch,
}, 'checkpoint.pt')

# 加载
ckpt = torch.load('checkpoint.pt')
model.load_state_dict(ckpt['model_state'])
optimizer.load_state_dict(ckpt['optimizer_state'])
```

### Q: 多 GPU 训练？
```python
model = torch.nn.DataParallel(model)
# 或者用 DistributedDataParallel
```

---

## 10. 完整训练脚本模板

见 `train_template.py`（同目录下）。
