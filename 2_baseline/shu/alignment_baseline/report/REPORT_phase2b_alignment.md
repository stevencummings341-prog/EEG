---
title: "Phase 2b Alignment — Canonical Report"
tags:
  - "#modality/eeg"
  - "#pipeline/4_analysis"
created: "2026-06-11"
updated: "2026-06-11"
status: "active"
---

# Phase 2b Alignment — Canonical Report

## 1. Core conclusion

- 无学习统计对齐**不足**：没有方法达到 +0.02 成功线。
- 最佳方法 `session_zscore` 仅 Δacc +0.0142；4/5 个方法为净正向。
- 这是有价值的 negative/diagnostic 结果，客观支持后续学习型适配（但本阶段不实现）。

## 2. Goal

检验无监督、纯统计的 test-time 对齐（不使用 target label、不在 target 上学习权重）
能否回收跨 session 的准确率下降。

## 3. Method

- `none_reference`（无对齐，来自 baseline_v1）/ `session_zscore` / `euclidean_alignment` /
  `riemannian_alignment`（log-Euclidean SPD 均值）/ `bn_statistics_adaptation` /
  `filterbank_reweighting`。
- 对齐统计量只用 source train 或 target 的无标签 X；BN 方法只刷新 running stats，无 optimizer.step。

## 4. Protocol

- single-source：ses-i → ses-j（both ok），每个 3-ok 被试 6 个方向。
- multi-source：ses-01+ses-02 → ses-03。
- 铁律：`y_test` 只用于最终评估，绝不进入训练/验证/早停/方法选择。

## 5. Results

| method | mean Δacc vs none |
|:---|---:|
| `session_zscore` | +0.0142 |
| `euclidean_alignment` | +0.0044 |
| `riemannian_alignment` | +0.0074 |
| `bn_statistics_adaptation` | +0.0056 |
| `filterbank_reweighting` | -0.0147 |

## 6. Analysis

- BN-stats 仅小幅正向；协方差对齐（EA/RA）略有害；z-score/filterbank 近中性。
- 按漂移等级：high-drift 被试受益最小（详见 `alignment_gain_by_drift_level.csv`）。
- 结论：纯统计对齐无法闭合跨 session gap，需要学习型 target 适配。

## 7. Relationship to previous phases

- 承接 Phase 1/2a：cross-session 仍有残余 gap。
- 支撑 Phase 2c：负结果指向 task representation / prototype drift 假设。

## 8. Next step

- Phase 2c Prototype Drift Analysis：验证掉点是否来自 embedding/prototype 漂移。
- 学习型 Step-3 适配（online/adapter/prototype/memory）为 future，本阶段不运行。

## 9. File list

- `tables/alignment_by_direction.csv`
- `tables/alignment_by_method.csv`
- `tables/alignment_by_model.csv`
- `tables/alignment_by_protocol.csv`
- `tables/alignment_by_subject.csv`
- `tables/alignment_gain_by_drift_level.csv`
- `tables/alignment_vs_baseline.csv`
- `tables/results_alignment_all.csv`
- `tables/run_status.csv`
- 详细原始报告：`../ALIGNMENT_BASELINE_REPORT.md`（13 节，含逐方向/逐被试/逐漂移等级）。
