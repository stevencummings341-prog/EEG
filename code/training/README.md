---
title: "Training Utilities"
tags:
  - "#modality/eeg"
created: "2026-06-10"
updated: "2026-08-04"
status: "active"
---

# Training Utilities

## 1. Purpose

通用训练和预测逻辑。两个训练器，用途不重叠：

| 文件 | 服务对象 | 特点 |
|:---|:---|:---|
| `trainer.py` | Phase 0–2c（已完成结果） | 交叉熵 + val-loss 早停 + 恢复最好权重；**冻结不动**，保证历史结果可复现。`predict()` 被两个训练器共用。 |
| `e2e_trainer.py` | 端到端基础模型主线（跨被试） | 支持非 CE 损失（DualCD 的钩子）、**每 cell 只存 `best.pt` + `last.pt`**、**断点续跑**（optimizer/scheduler/RNG/history 全存 `last.pt`，原子写入）、cosine schedule、梯度裁剪、可选 AMP。 |

## 2. What Belongs Here

通用训练/预测/优化器构造。数据划分与协议逻辑属于 `code/experiments/`，不要放这里。

## 3. Update Rules

保持 model-agnostic；不要为单个模型改 trainer 造成不公平——模型特有的训练步骤通过
可选钩子（`uses_custom_loss` / `training_step` / `after_optimizer_step`）表达。

**不要给 `trainer.py` 加行为**：Phase 0–2c 的数字是用它跑出来的。新需求进 `e2e_trainer.py`。

## 4. Related Files

- `AGENTS.md`: 唯一权威灵魂记忆。
- `0_docs/STRUCTURE_AND_FILE_GUIDE.md`: 全项目结构和文件职责说明。
- `0_docs/FILE_CATALOG.md`: 新增文件索引。
