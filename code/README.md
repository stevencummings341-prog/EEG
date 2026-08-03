---
title: "Code Framework"
tags:
  - "#modality/eeg"
created: "2026-06-10"
updated: "2026-06-10"
status: "active"
---

# Code Framework

## 1. Purpose

模块化代码框架（数据集/模型/方法/实验分层）。人工入口是 `code/run.py`。

## 2. What Belongs Here

- 与本目录职责直接相关的文件。
- 能帮助后来的人理解实验、代码或结果的索引说明。
- 必要时放 README、manifest、报告或轻量配置；大型数据和 checkpoint 不放在文档目录。

## 3. Update Rules

新增数据集/模型/方法/实验必须落在对应层，同时更新 config、README、FILE_CATALOG。

## 4. Related Files

- `AGENTS.md`: 唯一权威灵魂记忆。
- `0_docs/STRUCTURE_AND_FILE_GUIDE.md`: 全项目结构和文件职责说明。
- `0_docs/FILE_CATALOG.md`: 新增文件索引。
