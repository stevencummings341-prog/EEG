---
title: "File Catalog"
tags:
  - "#modality/eeg"
created: "2026-06-10"
updated: "2026-06-10"
status: "active"
---

# File Catalog

新增 source/config/doc/结果文件时更新本表。

## 根目录

| 路径 | 作用 |
|:---|:---|
| `AGENTS.md` | 唯一权威灵魂记忆。 |
| `README.md` | 人类入口与结构总览。 |
| `proposal.md` | 项目提案。 |
| `progress.md` | 进度日记（PROGRESS 角色，逐条追加）。 |
| `experiment_log.md` | 实验日志速查。 |
| `results.md` | 结果速查表。 |

## 0_docs

| 路径 | 作用 |
|:---|:---|
| `0_docs/ARCHITECTURE.md` | 结构 + 代码分层 + 文件职责。 |
| `0_docs/STATUS.md` | 进度 + 运行就绪 + SHU 就绪 + 下一步 + 清理策略。 |
| `0_docs/FILE_CATALOG.md` | 本文件，文件索引。 |
| `0_docs/operation_log.md` | 文件系统操作日志。 |

## 阶段结果

| 路径 | 作用 |
|:---|:---|
| `1_session_drift/{report,tables,figures}/` | Phase 0 漂移诊断结果。 |
| `2_baseline/no_alignment_baseline/{report,tables,figures}/` | Phase 1 baseline + Phase 2a multi-source 结果。 |
| `2_baseline/alignment_baseline/{report,tables,figures}/` | Phase 2b alignment 结果。 |
| `4_experiments/` | Phase 2c+ 新实验（待建）。 |
| `5_papers/` | 论文/汇报材料。 |

## 代码

| 路径 | 作用 |
|:---|:---|
| `code/run.py` | 统一入口：dry-run / 训练 / `--summarize`，进程内调度。 |
| `code/runners.py` | Phase 0/1/2a/2b 的进程内训练 runner（直连 code/ 模块）。 |
| `code/summaries/session.py` | Phase 1 baseline 汇总（表/图/原始报告）。 |
| `code/summaries/multisource.py` | Phase 2a multi-source 汇总。 |
| `code/summaries/alignment.py` | Phase 2b alignment 汇总。 |
| `code/summaries/canonical.py` | 按 9 段结构从汇总 CSV 生成 canonical `REPORT.md`。 |
| `code/summaries/summarize.py` | `--summarize` 调度器：原始汇总 + canonical。 |
| `code/configs/datasets/*.yaml` | 数据集配置（wbci_shu / shu）。 |
| `code/configs/models/*.yaml` | 模型超参配置。 |
| `code/configs/experiments/*.yaml` | 阶段实验配置。 |
| `code/datasets/` | 数据集适配器与 split/通道映射。 |
| `code/models/` | 模型与 registry。 |
| `code/methods/` | 对齐与适应方法。 |
| `code/experiments/` | 漂移/baseline/multi-source/alignment 协议 + 指标。 |
| `code/training/` | 通用 trainer。 |
| `code/preprocessing/` | 预处理逻辑。 |
| `code/utils/` | config/io/logging/paths/seed。 |

## backup

| 路径 | 作用 |
|:---|:---|
| `backup/README.md` | backup 索引与策略。 |
| `backup/COMPLETED_ARTIFACTS_INDEX.md` | 历史结果/权重/日志归档位置。 |
| `backup/root_archive_2026-06-10/` | 清理前的旧代码/文档/产物/权重/日志。 |
| `backup/legacy_snapshot_2026-06-10/` | 更早的轻量快照。 |
