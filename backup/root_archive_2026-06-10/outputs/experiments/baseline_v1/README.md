---
title: "Baseline V1 Outputs"
tags:
  - "#pipeline/4_analysis"
  - "#modality/eeg"
created: "2026-06-10"
updated: "2026-06-10"
status: "active"
---

# Baseline V1 Outputs

## 1. Purpose

该目录是 Phase 1 baseline 与 multi-source 规范化阅读入口之一，包含已完成结果的表、图、报告或 provenance。

## 2. Contents

- `runs/`: per-run CSV or metadata when present.
- `splits/`: exact data split JSON when present.
- `tables/`: aggregated CSV summaries.
- `figures/`: generated figures.
- `*_REPORT.md` / `RUN_STATUS.md`: human-readable reports and completion status.

## 3. Rules

不要覆盖 baseline_v1；如需复跑，创建 baseline_v2 或新的 run_id。

## 4. Related Indexes

- `backup/COMPLETED_ARTIFACTS_INDEX.md` explains why completed outputs stay in place.
- `0_docs/RUN_READINESS_AND_CLEANUP_GUIDE.md` explains cleanup and deletion policy.
