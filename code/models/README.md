---
title: "Model Layer"
tags:
  - "#modality/eeg"
created: "2026-06-10"
updated: "2026-06-10"
status: "active"
---

# Model Layer

## 1. Purpose

模型定义层，所有模型共享 `{logits, features, confidence}` 输出契约。

## 2. What Belongs Here

- 与本目录职责直接相关的文件。
- 能帮助后来的人理解实验、代码或结果的索引说明。
- 必要时放 README、manifest、报告或轻量配置；大型数据和 checkpoint 不放在文档目录。

## 3. Update Rules

新增模型要注册到 registry，并给出对应 `code/configs/models/*.yaml`。

## 4. Related Files

- `AGENTS.md`: 唯一权威灵魂记忆。
- `0_docs/STRUCTURE_AND_FILE_GUIDE.md`: 全项目结构和文件职责说明。
- `0_docs/FILE_CATALOG.md`: 新增文件索引。
