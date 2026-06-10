---
title: "Dataset Layer"
tags:
  - "#modality/eeg"
created: "2026-06-10"
updated: "2026-06-10"
status: "active"
---

# Dataset Layer

## 1. Purpose

统一数据集适配层，负责 Session 抽象、manifest/session 索引、通道对齐。

## 2. What Belongs Here

- 与本目录职责直接相关的文件。
- 能帮助后来的人理解实验、代码或结果的索引说明。
- 必要时放 README、manifest、报告或轻量配置；大型数据和 checkpoint 不放在文档目录。

## 3. Update Rules

不要在 `load()` 中偷偷做重预处理；raw-to-npz 必须显式实验执行。

## 4. Related Files

- `AGENTS.md`: 唯一权威灵魂记忆。
- `0_docs/STRUCTURE_AND_FILE_GUIDE.md`: 全项目结构和文件职责说明。
- `0_docs/FILE_CATALOG.md`: 新增文件索引。
