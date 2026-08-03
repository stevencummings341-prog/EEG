---
title: "Phase 1 Baseline — Canonical Report"
tags:
  - "#modality/eeg"
  - "#pipeline/4_analysis"
created: "2026-06-11"
updated: "2026-06-11"
status: "active"
---

# Phase 1 Baseline — Canonical Report

## 1. Core conclusion

- 跨 session 存在明显性能下降：`deepconvnet` within 0.606 → cross 0.536 (drop 0.070)；`eegnet` within 0.611 → cross 0.538 (drop 0.073)；`fbcnet` within 0.553 → cross 0.508 (drop 0.046)。
- within-session 是上界，single-source cross-session 量化了漂移代价。

## 2. Goal

建立统一、公平、无泄漏的 within-session 与 single-source cross-session baseline，
量化同一被试跨 session 的解码性能下降，作为后续对齐/适应方法的对照基准。

## 3. Method

- 模型：EEGNet / DeepConvNet / FBCNet，统一 `{logits, features, confidence}` 契约。
- 统一 trainer（CE + 早停）、统一 metrics（acc/bacc/macro-F1/AUC/NLL/Brier/ECE）。
- 多 seed，报告 mean ± std across seeds。

## 4. Protocol

- within-session：每个 ok session 内 StratifiedKFold；val 仅从 train 切出。
- cross-session：同被试有向 session 对 train ses-i → test ses-j（both ok）。
- 数据入口：`eog_ecg_clean` 的 `status=ok` 148 sessions；不用 derivatives .mat。
- 无泄漏：test session 的 label 绝不进入 train/val/早停。

## 5. Results

| model | within Acc | cross Acc | drop |
|:---|---:|---:|---:|
| `deepconvnet` | 0.606±0.004 | 0.536±0.003 | 0.070 |
| `eegnet` | 0.611±0.004 | 0.538±0.005 | 0.073 |
| `fbcnet` | 0.553±0.006 | 0.508±0.001 | 0.046 |

## 6. Analysis

- 与论文 within-session 10-fold 对照（%）：eegnet 论文 85.32，deepconvnet 论文 84.47，fbcnet 论文 78.4。
- 跨 session 下降主要由分布漂移导致（见 Phase 0 漂移诊断：空间模式 + μ/β 频谱）。
- 排序与论文趋势一致，within<cross 的结论稳健。

## 7. Relationship to previous phases

- 承接 Phase 0：漂移诊断解释了为何 cross-session 会掉点。
- 支撑 Phase 2a：multi-source 训练是否能回收部分跨 session gap。

## 8. Next step

- Phase 2a：multi-source ses-01+02 → ses-03。
- Phase 2b：no-learning alignment baseline。

## 9. File list

- `SESSION_MODEL_COMPARE_REPORT.md`
- `cross_by_direction.csv`
- `cross_by_seed.csv`
- `cross_session_accuracy_matrix_by_model.png`
- `model_ranking.md`
- `protocol_comparison.png`
- `results_cross_session.csv`
- `results_within_session.csv`
- `summary_by_model_protocol.csv`
- `within_by_seed.csv`
- `within_session_accuracy_boxplot.png`
- `within_session_wise.csv`
- 详细原始报告：`SESSION_MODEL_COMPARE_REPORT.md`（同目录，含可靠性检查与逐方向表）。
