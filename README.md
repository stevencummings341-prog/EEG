---
title: "EEG-MI Online 项目入口"
tags:
  - "#pipeline/5_dl_model"
  - "#modality/eeg"
  - "#method/domain_generalization"
created: "2026-06-10"
updated: "2026-06-10"
status: "active"
---

# EEG-MI Online

> 多数据集运动想象 EEG 跨 session 泛化研究：WBCIC-SHU 已完成主线，SHU 2022 为新增对照数据集。

## 0. 当前定位

已完成 Phase 0 漂移诊断、Phase 1 baseline、Phase 2a multi-source、Phase 2b no-learning alignment。结论：无学习统计对齐不足，下一步进入 Phase 2c Prototype Drift Analysis。未运行的内容不写成已完成。

## 1. 目录结构

```text
eeg-mi-online/
├── 0_docs/                 # 文档中心（ARCHITECTURE / STATUS / FILE_CATALOG / operation_log）
├── 1_session_drift/        # Phase 0 漂移诊断结果
├── 2_baseline/             # Phase 1/2a baseline + 2b alignment 结果
├── 3_online_adaptation/    # 在线适应，future
├── 4_experiments/          # Phase 2c+ 新实验入口
├── 5_papers/               # 论文与汇报材料
├── backup/                 # 旧代码/文档/历史产物/权重/日志归档
├── code/                   # 代码框架，唯一人工入口 code/run.py
├── inbox/                  # 临时交接材料
├── AGENTS.md               # 唯一灵魂记忆（先读）
├── proposal.md             # 项目提案
├── progress.md             # 进度日记（PROGRESS 角色）
├── experiment_log.md       # 实验日志速查
└── results.md              # 结果速查
```

## 2. 运行入口

```bash
cd /share/home/yuan/SYX/eeg-mi-online
python code/run.py --dry-run --config code/configs/experiments/phase1_baseline.yaml
```

完整训练当前被拦截（旧脚本已归档进 backup）；重训练需先让 `code/run.py` 直连 `code/experiments`，或临时从 backup 恢复兼容层。GPU 任务走 Slurm + `mi_torch_cu118`，先 smoke 再 full。

## 3. 必读文档

| 文档 | 用途 |
|:---|:---|
| `AGENTS.md` | 唯一权威灵魂记忆。 |
| `progress.md` | 进度日记。 |
| `0_docs/ARCHITECTURE.md` | 目录结构 + 代码分层 + 文件职责。 |
| `0_docs/STATUS.md` | 进度、能否跑、SHU 就绪、下一步、清理策略。 |
| `0_docs/FILE_CATALOG.md` | 文件索引。 |

## 4. 数据边界

外部数据只读：WBCIC-SHU `/share/workspace2/moto_imagination/WBCIC_SHU`，SHU `/share/workspace2/moto_imagination/SHU`。本仓库只写项目内文件，不修改外部原始数据。
