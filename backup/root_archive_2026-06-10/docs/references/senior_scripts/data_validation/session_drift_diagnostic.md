---
title: "Session 漂移诊断脚本说明"
tags:
  - "#pipeline/4_analysis"
  - "#method/domain_generalization"
created: 2026-06-06
---

# Session 漂移诊断脚本

## 概述

量化跨 session EEG 分布漂移，回答"跨 session 泛化为什么困难"。

## 使用方式

```bash
python session_drift_diagnostic.py --data_dir /path/to/eog_ecg_clean/ --output_dir ./drift_report
```

### 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `--data_dir` | 是 | -- | 预处理 .npz 文件目录 |
| `--output_dir` | 否 | `./drift_report` | 输出目录 |

### 数据格式要求

每个 `.npz` 文件包含：
- `X`: ndarray, shape `[trials, channels, timepoints]`
- `y`: ndarray, shape `[trials]`（标签值 1=左手, 2=右手）

文件名格式支持：`sub-XXX_ses-YY.npz`, `subXXX_sesYY.npz`, `SXX_SessionY.npz`

### 输出文件

| 文件 | 说明 |
|:---|:---|
| `session_drift_report.csv` | 漂移指标表（每个 session pair 一行） |
| `figures/distribution_distance_heatmap.png` | MMD/CORAL 热力图 |
| `figures/band_power_shift_hist.png` | μ/β 功率漂移分布 |
| `figures/erd_ers_correlation_hist.png` | ERD/ERS 模式一致性 |
| `figures/csp_similarity_hist.png` | CSP 模式相似度分布 |
| `figures/fisher_ratio_scatter.png` | Fisher 判别比漂移 |
| `figures/rms_ratio_hist.png` | 信号幅值一致性 |
| `figures/metric_correlation_matrix.png` | 指标相关性矩阵 |
| `figures/session_pair_comparison.png` | 不同 session 对的指标对比 |

## 计算的指标

| 指标 | 说明 | 计算方法 |
|:---|:---|:---|
| `mmd` | MMD 距离 | RBF kernel, 中位数启发式 γ |
| `coral` | CORAL 距离 | 协方差矩阵 Frobenius 范数 |
| `mu_power_shift` | μ 功率漂移 | log₁₀ 功率差 (8-13 Hz) |
| `beta_power_shift` | β 功率漂移 | log₁₀ 功率差 (13-30 Hz) |
| `mu_ks_stat` | μ KS 统计量 | 两样本 Kolmogorov-Smirnov 检验 |
| `erd_mu_corr` | μ ERD/ERS 相关 | 空间模式 Pearson 相关系数 |
| `erd_beta_corr` | β ERD/ERS 相关 | 同上 |
| `csp_similarity` | CSP 相似度 | 空间滤波器余弦相似度 |
| `rms_ratio_median` | RMS 比值中位数 | 通道级幅值一致性 |
| `fisher_i/j` | Fisher 判别比 | 类间/类内方差比 |
| `fisher_shift` | Fisher 漂移 | session j - session i |

## 依赖

```
numpy >= 1.21
scipy >= 1.7
scikit-learn >= 1.0
matplotlib >= 3.5
seaborn >= 0.12
pandas >= 1.4
```

## 在服务器上运行

```bash
# 1. 将脚本 scp 到服务器
scp session_drift_diagnostic.py user@server:/path/to/workdir/

# 2. SSH 登录
ssh user@server

# 3. 运行
python session_drift_diagnostic.py \
    --data_dir /share/workspace2/moto_imagination/WBCIC_SHU/processed/eog_ecg_clean/ \
    --output_dir ./drift_report
```
