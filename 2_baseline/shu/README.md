---
title: "2_baseline · shu · Phase 1/2a/2b Baseline 与 Alignment"
tags:
  - "#modality/eeg"
  - "#method/domain_generalization"
created: "2026-06-11"
updated: "2026-06-11"
status: "active"
---

# 2_baseline — shu

数据集: SHU 2022 (25 人 × 5 天 × 32ch, 2C 左右手抓握)

本目录是 **Phase 1/2a/2b Baseline 与 Alignment** 在 shu 上的结果区。

## 实验

- `no_alignment_baseline/`: Phase 1 within/cross + Phase 2a multi-source (无对齐)
- `alignment_baseline/`: Phase 2b no-learning alignment (有对齐)

## 约定

- 每个实验下按 `report/ tables/ figures/` 分层。
- 已完成结果只读；复跑用新 `run_id`，不覆盖历史。
- 命名规范见 `AGENTS.md` / `CLAUDE.md` 第 9 节。
