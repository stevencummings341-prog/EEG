---
title: "Model Layer"
tags:
  - "#modality/eeg"
created: "2026-06-10"
updated: "2026-08-04"
status: "active"
---

# Model Layer

## 1. Purpose

模型定义层，所有模型共享 `{logits, features, confidence}` 输出契约，输入统一 `[B, C, T]`。

## 2. What Belongs Here

| 文件/目录 | 内容 |
|:---|:---|
| `registry.py` | 唯一构建入口 `build_model(name, n_channels, n_times, n_classes, sfreq, params)`。 |
| `eegnet.py` / `deepconvnet.py` / `fbcnet.py` | Phase 0–2c 的三个基线（`BASELINE_MODEL_NAMES`）。 |
| `cap_eegnet.py` | 本项目 v1 模型（含学习型 confidence 头；prototype/online 仍是 future）。 |
| `eeg_foundation/` | **端到端主线的 5 个 S4 / DINO-DualCD 模型**（学长包移植 + 项目契约包装）。见该目录 README。 |

## 3. Update Rules

新增模型要注册到 registry，并给出对应 `code/configs/models/*.yaml`。

`eeg_foundation/` 的 DualCD 变体额外暴露三个**可选**训练钩子
（`uses_custom_loss` / `training_step` / `after_optimizer_step`），只有
`code/training/e2e_trainer.py` 会用；没有这些钩子的模型走普通交叉熵，所以基线模型也能跑同一个
训练器。**不要为了某个模型改公共 trainer**，用钩子。

## 4. Related Files

- `AGENTS.md`: 唯一权威灵魂记忆。
- `0_docs/STRUCTURE_AND_FILE_GUIDE.md`: 全项目结构和文件职责说明。
- `0_docs/FILE_CATALOG.md`: 新增文件索引。
