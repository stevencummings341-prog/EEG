---
title: "Experiment Configs"
tags:
  - "#modality/eeg"
created: "2026-06-10"
updated: "2026-08-04"
status: "active"
---

# Experiment Configs

## 1. Purpose

每个 phase 的可运行配置。`experiment.name` 决定用哪个 runner（见 `code/runners.py` 的
`PHASE_RUNNERS`），所以两个数据集可以共用同一个 runner，只换数据与输出路径。

## 2. What Belongs Here

| 配置 | 轴 | runner |
|:---|:---|:---|
| `phase0_drift_diagnostic` / `shu_phase0_drift_diagnostic` | 跨 session 漂移诊断 | `phase0_drift_diagnostic` |
| `phase1_baseline` / `shu_phase1_baseline` | within-session CV + 单源跨 session | `phase1_baseline` |
| `phase2a_multisource` / `shu_phase2a_multisource` | 多源跨 session | `phase2a_multisource` |
| `phase2b_alignment` / `shu_phase2b_alignment` | 无学习对齐 | `phase2b_alignment` |
| `phase2c_prototype_drift` / `shu_phase2c_prototype_drift` | 原型漂移诊断 | `phase2c_prototype_drift` |
| `phase3_tta` / `shu_phase3_tta` / `phase3_tta_full_a0`(opt-in) | 跨 session TTA（paused） | `phase3_tta` |
| **`foundation_cross_subject` / `shu_foundation_cross_subject`** | **跨被试端到端（当前主线）** | `foundation_cross_subject` |

## 3. Update Rules

新增实验配置必须说明 dataset/model/method/protocol/train/output，并指定唯一 run_id。

**两个数据集永远分开跑**：WBCIC 58ch、SHU 32ch，各自一个 config、各自的
`outputs/experiments/<dataset>/<run_id>/` 与 `checkpoints/<dataset>/<run_id>/`。

协议参数尚未获批时，在文件头明确写 `⚠ PROTOCOL STATUS: pending_advisor_confirmation` 并指向
对应备忘，避免后来的人把默认值当成已确认协议。

## 4. Related Files

- `AGENTS.md`: 唯一权威灵魂记忆。
- `0_docs/STRUCTURE_AND_FILE_GUIDE.md`: 全项目结构和文件职责说明。
- `0_docs/FILE_CATALOG.md`: 新增文件索引。
