---
title: "Experiment Outputs"
tags:
  - "#pipeline/4_analysis"
  - "#modality/eeg"
created: "2026-06-10"
updated: "2026-06-10"
status: "active"
---

# Experiment Outputs

## 1. Purpose

该目录存放实验输出的历史产物和规范化结果。它不是代码入口，不要手工修改 run CSV、split JSON 或 checkpoint 路径。

## 2. Contents

- `runs/`: per-run CSV or metadata when present.
- `splits/`: exact data split JSON when present.
- `tables/`: aggregated CSV summaries.
- `figures/`: generated figures.
- `*_REPORT.md` / `RUN_STATUS.md`: human-readable reports and completion status.

## 3. Rules

新增实验应使用唯一 run_id 子目录；完成后写报告、tables、figures、manifest_sources 或 RUN_STATUS。

## 4. Related Indexes

- `backup/COMPLETED_ARTIFACTS_INDEX.md` explains why completed outputs stay in place.
- `0_docs/RUN_READINESS_AND_CLEANUP_GUIDE.md` explains cleanup and deletion policy.
