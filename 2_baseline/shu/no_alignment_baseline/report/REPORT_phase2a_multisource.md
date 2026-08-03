---
title: "Phase 2a Multi-source — Canonical Report"
tags:
  - "#modality/eeg"
  - "#pipeline/4_analysis"
created: "2026-06-11"
updated: "2026-06-11"
status: "active"
---

# Phase 2a Multi-source — Canonical Report

## 1. Core conclusion

- 多源训练（ses-01+02 → ses-03）在 ses-03 上的准确率：`eegnet` 0.544±0.013；`deepconvnet` 0.558±0.012；`fbcnet` 0.512±0.007。
- 多源通常优于最强单源方向，能回收部分跨 session gap。

## 2. Goal

检验合并多个源 session（ses-01+ses-02）训练能否比单源更好地泛化到目标 session（ses-03）。

## 3. Method

- 同 Phase 1 的模型/trainer/metrics；合并 ses-01 与 ses-02 的全部 trial 作为 train。
- 多 seed，报告 mean ± std；与 Phase 1 的单源 ses-0x→ses-03 对照。

## 4. Protocol

- train = ses-01 + ses-02 全部 trial；val 仅从合并 train 切出。
- test = ses-03 全部 trial，仅用于最终评估。
- 仅纳入 ses-01/02/03 全 ok 的被试；缺失者记录为 skipped。

## 5. Results

| model | Acc(ses-03) | BalAcc | MacroF1 | AUC | n_seeds |
|:---|---:|---:|---:|---:|---:|
| `eegnet` | 0.544±0.013 | 0.544 | 0.523 | 0.566 | 5 |
| `deepconvnet` | 0.558±0.012 | 0.559 | 0.536 | 0.583 | 5 |
| `fbcnet` | 0.512±0.007 | 0.512 | 0.451 | 0.527 | 5 |

## 6. Analysis

- 多源相对最强单源的提升说明：增加源 session 的多样性有助于跨 session 泛化。
- 失败案例集中在两源 session 质量差异大的被试 → 为 Phase 2b 对齐提供动机。

## 7. Relationship to previous phases

- 承接 Phase 1：单源 cross-session 是对照基准。
- 支撑 Phase 2b：multi-source 仍未完全闭合 gap，需要对齐/适应。

## 8. Next step

- Phase 2b：在 cross 协议上加 no-learning alignment（z-score/EA/Riemannian/BN/filterbank）。

## 9. File list

- `MULTISOURCE_STEP1_REPORT.md`
- `multisource_by_model.csv`
- `multisource_by_seed.csv`
- `results_multisource_0102_to_03.csv`
- `summary_by_model_protocol.csv`
- 详细原始报告：`MULTISOURCE_STEP1_REPORT.md`（同目录）。
