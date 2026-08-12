---
title: "Experiment Layer"
tags:
  - "#modality/eeg"
created: "2026-06-10"
updated: "2026-08-07"
status: "active"
---

# Experiment Layer

## 1. Purpose

实验协议层，组合 dataset、model、method、trainer、metrics。

## 2. What Belongs Here

| 文件 | 泛化轴 | 划分单位 |
|:---|:---|:---|
| `session_protocols.py` | within-session CV + 单源跨 session | trial（同 session 内）/ session |
| `session_multisource_protocols.py` | Phase 2a 多源跨 session | session |
| `session_alignment_protocols.py` | Phase 2b 无学习对齐 | session |
| `prototype_drift.py` / `prototype_drift_summarize.py` | Phase 2c 原型漂移诊断 | session |
| `session_tta.py` | Phase 3 TTA（paused，未废弃） | session |
| `cross_subject_protocols.py` | **跨被试（当前主线）** | **subject**（LOSO / subject k-fold / holdout）；可选 session 级 train/val |
| `metrics.py` / `data_quality.py` | 共用指标与质检 | — |

## 3. Update Rules

新增协议必须说明数据划分、防泄漏规则、输出 CSV 字段。

跨被试协议的三条铁律（`cross_subject_protocols.py` 里有断言 + `tests/foundation/` 有对应测试）：
测试被试绝不出现在 train/val；验证集来自留出训练被试（`val_mode=subjects`）、训练被试的 trial
切片（`val_mode=trials`），或**同批非测试被试按 session 切**（`val_mode=sessions`：train
ses-01+02 / val ses-03，对齐 SHUv5 / WBCIC-SHU 3C 论文）；归一化必须是 per-trial fit-free 的。

**三曲线（2026-08-10）**：`train.curves: true` 时，协议层额外构建两个**只做监控**的 loader
（训练集固定子集在 eval 模式 + 测试被试），交给 `e2e_trainer` 每 epoch 记录进 `history` 的
`train_eval` / `test`。**test 绝不参与模型选择**——`best.pt`、早停、`best_score` 只读 val，
守卫在 `code/training/e2e_trainer.py`。绘图见 `scripts/plot_three_curves.py`。

## 4. Related Files

- `AGENTS.md`: 唯一权威灵魂记忆。
- `0_docs/STRUCTURE_AND_FILE_GUIDE.md`: 全项目结构和文件职责说明。
- `0_docs/FILE_CATALOG.md`: 新增文件索引。
