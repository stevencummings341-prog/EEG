---
title: "SHU 2022 Phase 0 Cross-session Drift — AI Analysis"
tags:
  - "#pipeline/5_dl_model"
  - "#modality/eeg"
  - "#method/domain_generalization"
  - "#paradigm/motor_imagery"
created: "2026-06-11"
updated: "2026-06-11"
status: "active"
---

# SHU 2022 跨 Session 漂移诊断 — AI 深度分析

> 数据层面诊断，非模型训练。所有数字均从 `tables/` 的 CSV / `summary.json` 重新读取，未捏造。
> 数据集：SHU 2022（25 被试 × 5 session × 32ch，250 Hz，作者发布 per-session `.mat`，仅标签 {1,2}->{0,1} 归一化）。

## 0. 核心结论 (honest)

SHU 的跨 session 漂移与 WBCIC-SHU **同质**：主要体现在**空间判别模式 + μ/β 频谱分布**，而**不是幅值**，且**左右手可分性没有被系统性削弱**。

- 整体分布距离 **MMD = 0.356**（median 0.344，std 0.149，n=250）→ 中等偏强漂移。
- **CSP 相似度仅 0.344**（median 0.334）→ 空间判别方向跨 session 只保留约 1/3，是漂移最强的维度。
- **ERD/ERS μ/β 空间相关 0.527 / 0.532** → 感觉运动激活指纹漂移约 47%。
- **RMS 比值 median ≈ 1.03**（mean 1.32 被少数离群放大）→ 幅值基本稳定，漂移不是幅值缩放问题。
- **Fisher shift ≈ -0.0012 ≈ 0** → 可分性没有统一增强或减弱，掉点更可能来自模式漂移而非类别不可分。
- 被试分层：high 9 / moderate 8 / stable 8（三分位）。最漂移 sub-017/008/019，最稳定 sub-002/006/020。

与 WBCIC 对比（CLAUDE.md 记录：MMD 0.238、CSP 0.420、ERD μ 0.419）：SHU 的 **MMD 更大、CSP 更低**，即 SHU 的跨 session **空间漂移更严重**；但定性机制一致。**这支持把 WBCIC 主线诊断/对齐/prototype 流程原样迁移到 SHU**。

## 1. 实验目标

量化同一被试在不同天 session 之间的 EEG 分布漂移，判断 SHU 跨 session 解码困难的来源，并与 WBCIC 诊断结论比对，确认双数据集并列主线是否成立。

## 2. 方法定义

复用 WBCIC 同一 runner（`code/experiments/session_drift.py`），同一指标集：MMD(RBF)、CORAL、μ/β power shift、μ-KS、ERD/ERS 空间相关、CSP 相似度、RMS 比值、Fisher 可分性/shift。被试内**无向** session pair（i<j）；5 session ⇒ 每被试 C(5,2)=10 pair。per-subject `drift_score = mean(z(MMD)+z(μKS)−z(CSP)−z(ERD_mu)−z(ERD_beta))`，按三分位分 high/moderate/stable。

## 3. 实验协议

- 数据：`SHU/processed/npz_clean/processed_manifest.csv`，status=ok = **125 session / 25 被试**全 ok。
- pair 总数 **250**（25 × 10），每个 pair 类型 n=25（5 session 全 ok，无 partial）。
- seed = 0；mmd_subsample=100；csp_components=4；erd_baseline_ratio=0.25。
- 硬件：Slurm gpu2node 节点 CPU 运行（`mi_torch_cu118`），耗时 367s。不在登录节点跑。
- 不涉及 train/test、不产生 accuracy。

## 4. 结果

### 4.1 总体（250 pairs）

| 指标 | mean | median | std |
|:---|---:|---:|---:|
| mmd | 0.3556 | 0.3437 | 0.1489 |
| mu_ks_stat | 0.2784 | 0.2515 | 0.1760 |
| csp_similarity | 0.3438 | 0.3341 | 0.1130 |
| erd_mu_corr | 0.5267 | 0.5564 | 0.2455 |
| erd_beta_corr | 0.5318 | 0.5871 | 0.2647 |
| mu_power_shift | 0.0372 | 0.0290 | 0.5073 |
| beta_power_shift | -0.0044 | -0.0015 | 0.4996 |
| rms_ratio_median | 1.3157 | 1.0323 | 1.1775 |
| fisher_shift | -0.0012 | -0.0005 | 0.0120 |
| coral | ~1.4e-08 | ~2.6e-15 | ~7.1e-08 |

### 4.2 按 session pair（n=25 each）

| pair | mmd_mean | csp_sim_mean | erd_mu_corr_mean |
|:---|---:|---:|---:|
| 01-02 | 0.3478 | 0.3588 | 0.5153 |
| 01-03 | 0.3564 | 0.3719 | 0.5451 |
| 01-04 | 0.4127 | 0.3399 | 0.5106 |
| 01-05 | 0.3687 | 0.3260 | 0.4888 |
| 02-03 | 0.2840 | 0.3550 | 0.5599 |
| 02-04 | 0.3407 | 0.3605 | 0.5447 |
| 02-05 | 0.3838 | 0.3352 | 0.4635 |
| 03-04 | 0.3559 | 0.3396 | 0.5679 |
| 03-05 | 0.3509 | 0.3199 | 0.5110 |
| 04-05 | 0.3550 | 0.3310 | 0.5607 |

- MMD 最大：**01-04 (0.413)**；最小：**02-03 (0.284)**。
- CSP 最高（最稳定空间模式）：01-03 (0.372)；最低：03-05 (0.320)。

### 4.3 被试分层

high 9 / moderate 8 / stable 8。最漂移：sub-017 (drift_score 1.189, mean_mmd 0.436, csp 0.332)、sub-008 (0.930, mmd 0.502, csp 0.298)、sub-019 (0.809)。最稳定：sub-002 (-1.089)、sub-006 (-1.066, mmd 0.238, csp 0.410)、sub-020 (-1.012)。

## 5. 分析

1. **空间漂移是主轴**：CSP 0.344 是所有稳定性指标里最低的（远离 1），说明跨 session 判别用的空间滤波方向重组最严重；ERD μ/β 0.53 也只中等。
2. **幅值不是主因**：RMS median≈1.03，mean 被离群拉高到 1.32（少数 session 增益异常）。后续若做对齐，全局幅值归一化预计无效——与 WBCIC 结论一致。
3. **可分性未塌缩**：fisher_shift≈0 且 std 仅 0.012，说明左右手在新 session 仍可分，掉点来自"判别方向/激活模式漂移"而非"类别不可分"。这正是 prototype drift 假设要检验的机制。
4. **没有单调"距离越远漂移越大"**：01-04 MMD 最大但 01-05 反而回落，02-03（相邻）最小。提示 SHU 5 session 之间存在非单调的 session 特异性（可能与采集顺序/状态有关），cross-session 方向不对称值得在 Phase 1 验证。
5. **CORAL 退化**：近零均值导致 CORAL≈0，与 WBCIC 一样仅作参考，不作主证据。

## 6. 与已有结果的关系

| 指标 | WBCIC-SHU | SHU 2022 | 含义 |
|:---|---:|---:|:---|
| MMD | 0.238 | 0.356 | SHU 整体漂移更大 |
| CSP similarity | 0.420 | 0.344 | SHU 空间模式保留更少（漂移更重）|
| ERD μ corr | 0.419 | 0.527 | SHU μ 空间指纹相对更稳 |
| RMS ratio | ~0.99 | ~1.03 | 两者幅值都稳定 |

定性机制一致（空间+频谱漂移，幅值稳定，可分性不塌），定量上 SHU 空间漂移更严重。WBCIC 主线（baseline drop → 对齐不足 → prototype drift = scatter 膨胀）的迁移在数据层面有依据。

## 7. 下一步建议

1. **Phase 1 baseline**：用同 125 ok session 跑 within vs directed cross，量化 SHU 的 accuracy drop，验证非单调方向不对称是否复现（关注 02-03 易、01-04 难）。
2. **对齐方法**优先考虑空间对齐（EA/CORAL/Riemannian）+ filterbank，而非幅值归一化（Phase 2b）。
3. **prototype drift（Phase 2c）**：SHU 空间漂移更重，是检验 scatter 膨胀机制的更强样本。
4. 关注 high-drift 被试 sub-017/008/019，作为对齐/适应失败的代表。

## 8. 文件清单

- 报告：`1_session_drift/shu/report/SESSION_DRIFT_REPORT.md`、`DRIFT_PAIR_SUBJECT_REPORT.md`、`per_subject_drift_summary.md`、本文件 `AI_ANALYSIS.md`
- 表：`1_session_drift/shu/tables/session_drift_report.csv`（250 行）、`session_pair_summary.csv`、`per_subject_drift_summary.csv`、`summary.json`
- 图：`1_session_drift/shu/figures/`（14 张）
- 重计算源：`outputs/analysis/shu/session_drift_v1/`
