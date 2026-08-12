---
title: "Model Configs"
tags:
  - "#modality/eeg"
created: "2026-06-10"
updated: "2026-08-09"
status: "active"
---

# Model Configs

## 1. Purpose

模型结构超参配置。数据维度（`n_channels` / `n_times` / `n_classes` / `sfreq`）**不写在这里**，
由实验层注入。

## 2. What Belongs Here

| 文件 | 模型 |
|:---|:---|
| `eegnet.yaml` / `deepconvnet.yaml` / `fbcnet.yaml` | Phase 0–2c 三基线 |
| `cap_eegnet.yaml` | 本项目 v1 模型 |
| `s4erp.yaml` | S4 + flatten，纯监督（端到端主线里最便宜的，先用它验证流程） |
| `dualcd_s4_pos.yaml` | S4 + attention pooling + DINO/DualCD（~2.0M） |
| `dualcd_s4_timepatch.yaml` | S4 + 时间分箱 + DINO/DualCD（~3.3M，**分箱边界必须按 4s trial 设**） |
| `dualcd_s4_flatten.yaml` | S4 + flatten + DINO/DualCD（~65.8M） |
| `dualcd_transformer.yaml` | Transformer + flatten + DINO/DualCD（~66.8M，最慢） |
| `atcnet.yaml` | **ATCNet（Altaheri 2023）已发表基线**，对标 DSGNet 论文 Table II（Acc 0.6834）；参数为上游默认值，不调参 |

## 3. Update Rules

只放结构参数，不放实验协议；实验组合在 experiments config 中定义。

实验 config 的 `model_params.<name>` 会覆盖这里的值（两处都要改，别只改一处）。
`code/configs/models/*.yaml` 是「这个模型的标准超参」，实验 config 是「这次实验真正用的」。

## 4. Related Files

- `AGENTS.md`: 唯一权威灵魂记忆。
- `0_docs/STRUCTURE_AND_FILE_GUIDE.md`: 全项目结构和文件职责说明。
- `0_docs/FILE_CATALOG.md`: 新增文件索引。
