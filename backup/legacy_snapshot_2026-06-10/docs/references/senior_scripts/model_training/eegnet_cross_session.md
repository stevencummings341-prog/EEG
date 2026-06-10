---
title: "EEGNet Cross-Session Baseline 脚本说明"
tags:
  - "#pipeline/5_dl_model"
  - "#method/domain_generalization"
created: 2026-06-06
---

# EEGNet Cross-Session Baseline 脚本

## 概述

建立 MI 跨 session/跨被试分类基线，量化"难到什么程度"。实现 EEGNet 的三种评估协议。

## 使用方式

```bash
# Within-session 10-fold CV
python eegnet_cross_session.py --data_dir /path/to/data --protocol within

# Cross-session (train on ses-i, test on ses-j)
python eegnet_cross_session.py --data_dir /path/to/data --protocol cross

# LOSO (leave-one-subject-out)
python eegnet_cross_session.py --data_dir /path/to/data --protocol loso

# 全部运行
python eegnet_cross_session.py --data_dir /path/to/data --protocol all
```

### 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `--data_dir` | 是 | -- | 预处理 .npz 文件目录 |
| `--output_dir` | 否 | `./baseline_results` | 输出目录 |
| `--protocol` | 否 | `all` | 评估协议 (within/cross/loso/all) |
| `--n_epochs` | 否 | 100 | 训练 epoch 数 |
| `--batch_size` | 否 | 16 | batch size |
| `--device` | 否 | `auto` | 设备 (cpu/cuda/auto) |

### 三种评估协议

| 协议 | 训练集 | 测试集 | 回答的问题 |
|:---|:---|:---|:---|
| **Within-session** | 同 session 10-fold | 同 session | 上界：无漂移时的最佳性能 |
| **Cross-session** | Session i | Session j | 漂移导致的性能损失 |
| **LOSO** | 41 被试所有 session | 1 被试所有 session | 跨被试泛化能力 |

### 输出文件

| 文件 | 说明 |
|:---|:---|
| `results_within_session.csv` | Within-session 结果（每个 subject/session 一行） |
| `results_cross_session.csv` | Cross-session 结果（每个 subject/session-pair 一行） |
| `results_loso.csv` | LOSO 结果（每个 subject/session 一行） |
| `figures/within_session_accuracy.png` | 每个被试的 within-session 准确率 |
| `figures/cross_session_matrix.png` | Cross-session 准确率矩阵 |
| `figures/cross_session_per_subject.png` | 每个被试的 cross-session 平均准确率 |
| `figures/loso_accuracy.png` | LOSO 准确率分布 |
| `figures/protocol_comparison.png` | 三种协议对比 |

## EEGNet 模型

参照 Lawhern et al. (2018) 实现：
- Block 1: Temporal Conv → Depthwise Spatial Conv → ELU → AvgPool → Dropout
- Block 2: Separable Conv → ELU → AvgPool → Dropout
- Classifier: Linear

超参数（与 WBCIC-SHU 论文一致）：
- F1=8, D=2, F2=16
- kernel_length=64
- dropout=0.5
- Adam optimizer, lr=0.001
- batch_size=16

## 依赖

```
numpy >= 1.21
torch >= 1.12
scikit-learn >= 1.0
pandas >= 1.4
matplotlib >= 3.5
seaborn >= 0.12
```

## 在服务器上运行

```bash
# 1. 将脚本 scp 到服务器
scp eegnet_cross_session.py user@server:/path/to/workdir/

# 2. SSH 登录
ssh user@server

# 3. 运行（建议用 GPU）
python eegnet_cross_session.py \
    --data_dir /share/workspace2/moto_imagination/WBCIC_SHU/processed/eog_ecg_clean/ \
    --output_dir ./baseline_results \
    --protocol all \
    --device cuda
```

## 注意事项

- LOSO 需要训练 51 个模型，每个 100 epochs，**预计耗时较长**（GPU 上约 2-4 小时）
- 建议先跑 `--protocol within` 验证环境，再跑 `--protocol all`
- 标签自动重映射为 0-indexed，支持任意标签值
