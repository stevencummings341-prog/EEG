# 跨 Session 漂移诊断实验报告 (Cross-session Drift Diagnostic)

> 本报告基于已计算的 `session_drift_report.csv`（144 个被试内 session pair / 50 被试）重新整理，未重跑任何漂移计算。

## A. 实验目的

量化**同一被试**在不同 session（不同天采集）之间的 EEG 分布漂移，回答“为什么跨 session 运动想象解码会比同 session 困难”。这是**数据层面的诊断**，用来指导后续 baseline 与自适应方法的设计，**不是模型训练**。

## B. 实验安排

- **数据来源**：`eog_ecg_clean_v1`，仅用 `status=ok` 的 **148 个 session**（已排除 5 个 failed）。
- 每个被试理论上有 3 个 session：`ses-01`、`ses-02`、`ses-03`。
- 跨 session 漂移按**同一被试内部**的 3 种 pair 计算（无向，i<j）：
  1. `ses-01 vs ses-02`
  2. `ses-01 vs ses-03`
  3. `ses-02 vs ses-03`
- **为什么是 144 pairs（不是 153）**：
  - 若 51 个被试都完整：51 × C(3,2) = **153** pairs；
  - 实际有 **5 个 failed session** 被排除（sub-023/ses-01、sub-024/ses-02、sub-024/ses-03、sub-026/ses-01、sub-032/ses-02）；
  - `sub-024` 只剩 **1 个** ok session → **0 pair**；
  - `sub-023`、`sub-026`、`sub-032` 各只剩 **2 个** ok session → 各 **1 pair**；
  - 其余 47 个被试 3 session 全 ok → 各 3 pair；
  - 合计 47×3 + 3×1 = **144 pairs / 50 个 eligible 被试**（pair 分布：1-2=47、1-3=48、2-3=49）。
- **本实验不涉及** train/test split、不涉及 baseline accuracy，只做数据层面的 drift 诊断。

## C. 指标说明（测什么 / 方向）

| 指标 | 测什么 | 方向 |
|---|---|---|
| MMD (RBF) | 两 session trial 分布的整体距离 | 越大漂移越大 |
| CORAL | 协方差/二阶统计差异 | 当前实现基于近零均值，**易受退化影响，仅作参考、不作主证据** |
| μ/β power shift | MI 关键频段(8–13/13–30 Hz)功率变化(j−i) | 越偏离 0 漂移越大 |
| μ KS statistic | μ 频段功率分布的两样本 KS | 越大分布变化越大（0=相同）|
| ERD/ERS spatial corr | sensorimotor 激活空间模式跨 session 相关 | 越接近 1 越稳定 |
| CSP similarity | 空间判别滤波器方向一致性 | 越接近 1 越稳定 |
| RMS ratio | 通道幅值比值(j/i) | 越接近 1 幅值越稳定 |
| Fisher ratio / shift | 左右手 MI 可分性及其变化 | shift≈0=可分性无系统变化 |

## D. 总体结果（144 pairs）

| 指标 | mean | median | std |
|---|---|---|---|
| `mmd` | 0.2381 | 0.2043 | 0.1091 |
| `coral` | 0.0988 | 0.0000 | 0.7840 |
| `mu_power_shift` | 0.0313 | -0.0230 | 0.3602 |
| `beta_power_shift` | 0.0260 | -0.0148 | 0.2931 |
| `mu_ks_stat` | 0.2463 | 0.1950 | 0.1827 |
| `erd_mu_corr` | 0.4194 | 0.4279 | 0.2867 |
| `erd_beta_corr` | 0.4815 | 0.5244 | 0.2916 |
| `csp_similarity` | 0.4197 | 0.4319 | 0.0755 |
| `rms_ratio_median` | 1.1065 | 0.9921 | 0.7968 |
| `fisher_shift` | -0.0011 | -0.0012 | 0.0185 |

**解读：**
- MMD≈0.238：整体存在**中等**分布漂移。
- CSP similarity≈0.420：空间判别模式只**中等稳定**。
- ERD/ERS μ/β corr≈0.419/0.482：感觉运动节律的空间模式存在明显漂移。
- RMS ratio median≈0.992：整体幅值**不是**主要漂移来源。
- Fisher shift≈-0.0011：平均 MI 可分性**没有**统一增强或减弱。
- CORAL 中位数≈0（受近零均值退化影响），不作为主证据。

## E. 按 session pair 分组分析

| pair | n | MMD mean | MMD median | μ-KS mean | CSP sim mean | ERD μ corr | ERD β corr | RMS ratio median | Fisher shift mean |
|---|---|---|---|---|---|---|---|---|---|
| **01-02** | 47 | 0.237 | 0.204 | 0.275 | 0.411 | 0.384 | 0.507 | 0.946 | -0.0034 |
| **01-03** | 48 | 0.247 | 0.215 | 0.270 | 0.421 | 0.414 | 0.433 | 0.964 | -0.0018 |
| **02-03** | 49 | 0.230 | 0.197 | 0.196 | 0.427 | 0.459 | 0.504 | 1.041 | 0.0017 |

- **MMD 最大的 pair**：`01-03`。
- **1-3 vs 1-2**：1-3 的 MMD（0.247）大于 1-2（0.237）——即“相隔越远（1→3）漂移是否更大”。
- **2-3 是否更稳定**：2-3 的 MMD=0.230、CSP=0.427，相比 1-2（MMD=0.237、CSP=0.411）更稳定。
- **学习效应**：若“后期 session 更稳定/技能趋于一致”，应看到 2-3 漂移最小且 CSP/ERD 最高；本数据中该趋势**部分成立**，需结合 baseline 的 within/cross accuracy 进一步判断。

## F. 按被试分析

`drift_level` 由综合漂移分 `drift_score`（z(MMD)+z(μ-KS)−z(CSP)−z(ERD_mu)−z(ERD_beta) 的均值）按三分位划分。完整表见 `per_subject_drift_summary.csv` / `.md`。

### Top 10 high-drift subjects

| subject | pairs | mean_mmd | mean_mu_ks | mean_csp_sim | mean_erd_mu | drift_score |
|---|---|---|---|---|---|---|
| sub-020 | 01-02,01-03,02-03 | 0.402 | 0.545 | 0.359 | -0.086 | 1.874 |
| sub-013 | 01-02,01-03,02-03 | 0.335 | 0.457 | 0.314 | 0.157 | 1.321 |
| sub-029 | 01-02,01-03,02-03 | 0.395 | 0.519 | 0.409 | -0.170 | 1.303 |
| sub-031 | 01-02,01-03,02-03 | 0.471 | 0.598 | 0.371 | 0.156 | 1.286 |
| sub-038 | 01-02,01-03,02-03 | 0.203 | 0.407 | 0.340 | 0.099 | 0.994 |
| sub-014 | 01-02,01-03,02-03 | 0.259 | 0.326 | 0.322 | 0.348 | 0.736 |
| sub-032 | 01-03 | 0.151 | 0.400 | 0.368 | -0.074 | 0.709 |
| sub-030 | 01-02,01-03,02-03 | 0.270 | 0.664 | 0.392 | 0.443 | 0.682 |
| sub-027 | 01-02,01-03,02-03 | 0.333 | 0.289 | 0.377 | 0.384 | 0.669 |
| sub-015 | 01-02,01-03,02-03 | 0.215 | 0.325 | 0.317 | 0.519 | 0.518 |

### Top 10 stable subjects

| subject | pairs | mean_mmd | mean_mu_ks | mean_csp_sim | mean_erd_mu | drift_score |
|---|---|---|---|---|---|---|
| sub-005 | 01-02,01-03,02-03 | 0.143 | 0.173 | 0.464 | 0.793 | -1.084 |
| sub-039 | 01-02,01-03,02-03 | 0.249 | 0.154 | 0.492 | 0.817 | -0.957 |
| sub-016 | 01-02,01-03,02-03 | 0.183 | 0.131 | 0.485 | 0.592 | -0.900 |
| sub-050 | 01-02,01-03,02-03 | 0.202 | 0.084 | 0.397 | 0.812 | -0.749 |
| sub-044 | 01-02,01-03,02-03 | 0.174 | 0.198 | 0.521 | 0.583 | -0.713 |
| sub-043 | 01-02,01-03,02-03 | 0.133 | 0.081 | 0.427 | 0.741 | -0.704 |
| sub-040 | 01-02,01-03,02-03 | 0.160 | 0.115 | 0.483 | 0.466 | -0.665 |
| sub-035 | 01-02,01-03,02-03 | 0.141 | 0.046 | 0.365 | 0.716 | -0.547 |
| sub-049 | 01-02,01-03,02-03 | 0.275 | 0.126 | 0.447 | 0.669 | -0.541 |
| sub-047 | 01-02,01-03,02-03 | 0.184 | 0.155 | 0.376 | 0.715 | -0.524 |

### Partial subjects（failed session 导致 pair 不全）

| subject | failed session | 剩余 ok session | 可用 pair |
|---|---|---|---|
| sub-023 | ses-01 | ses-02, ses-03 | 02-03 |
| sub-024 | ses-02/ses-03 | 仅 ses-01 | （无 pair） |
| sub-026 | ses-01 | ses-02, ses-03 | 02-03 |
| sub-032 | ses-02 | ses-01, ses-03 | 01-03 |

- **`sub-024` 没有任何 pair**：它有 2 个 failed session（ses-02、ses-03），只剩 1 个 ok session（ses-01），无法构成任何被试内 session pair，因此不参与漂移统计。

## G. 图

- `figures/session_pair_metric_summary.png`
- `figures/subject_mmd_heatmap.png`
- `figures/subject_csp_heatmap.png`
- `figures/subject_erd_mu_heatmap.png`
- `figures/high_drift_subjects_bar.png`
- `figures/signal_quality_shift.png`
- `figures/distribution_distance_hist.png`
- `figures/band_power_shift_hist.png`
- `figures/erd_ers_correlation_hist.png`
- `figures/csp_similarity_hist.png`
- `figures/fisher_ratio_scatter.png`
- `figures/rms_ratio_hist.png`
- `figures/metric_correlation_matrix.png`
- `figures/session_pair_comparison.png`

## H. 结论

- 本数据集的跨 session 漂移**主要体现为空间模式与 μ/β 频谱分布的变化**：CSP 相似度仅约 0.420、ERD/ERS μ 空间相关约 0.419，且 μ-KS 约 0.246 表明频段功率分布发生了可观变化。
- 整体**幅值中位数接近稳定**（RMS 比值中位数≈0.992），因此跨 session 困难**不是简单的幅值缩放问题**，仅做全局幅值归一化不足以解决。
- **Fisher shift 平均≈-0.0011**：左右手可分性没有被一致增强或削弱，说明后续模型跨 session 性能下降**更可能来自模式漂移**，而非类别在新 session 完全不可分。
- **后续 baseline**：需重点比较 within-session 与 cross-session 的 accuracy drop，量化漂移代价。
- **后续自适应方法**：应优先考虑 **spatial alignment（Euclidean Alignment / CORAL）**、**frequency / filter-bank adaptation**、**BN/adapter** 等针对“空间+频谱模式漂移”的手段，而不是只做全局幅值归一化。

> 备注：当前 KS 仅计算了 μ 频段（`mu_ks_stat`）；β 频段 KS 需要重跑 drift（重新读取 npz），本次按要求**未重算**，可作为后续小补充。
