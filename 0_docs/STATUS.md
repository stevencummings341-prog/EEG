---
title: "Status and Readiness"
tags:
  - "#pipeline/5_dl_model"
  - "#modality/eeg"
created: "2026-06-10"
updated: "2026-06-10"
status: "active"
---

# Status and Readiness

> 当前进度 + 能否继续跑 + SHU 就绪度 + 下一步 + 冗余/删除策略。逐条进度记忆在根目录 `progress.md`。

## 1. 一句话现状

WBCIC-SHU 跨 session 主线已完成到 Phase 2b（no-learning alignment 不足）；仓库已重构为 P10 风格多数据集框架，历史结果已同步进阶段目录；下一步是 Phase 2c Prototype Drift Analysis。

## 2. 进度表

| 阶段 | 状态 | 结果位置 |
|:---|:---|:---|
| Phase 0 Drift Diagnostic | done | `1_session_drift/` |
| Phase 1 Baseline | done | `2_baseline/` |
| Phase 2a Multi-source | done | `2_baseline/cross_session/` |
| Phase 2b No-learning Alignment | done | `2_baseline/alignment_baseline/` |
| Phase 2c Prototype Drift | next, 未运行 | `4_experiments/`（待建） |
| Phase 3+ Adaptation/Online/Agent | future | `3_online_adaptation/` |

关键结论：漂移主要是空间模式 + μ/β 频谱；跨 session drop 约 10pp；多源优于最强单源；无学习对齐不足（BN-stats 小幅正收益，无方法过 +2pp）。

## 3. 现在能不能跑

可以。可训练能力已在新架构下恢复：

- `code/run.py` 通过 `code/runners.py` 在进程内直接调用 `code/` 模块，不依赖归档的旧 `scripts/`。
- Phase 0/1/2a/2b 的 runner 都已实现（`code/runners.py` 的 `PHASE_RUNNERS`）。
- 已用 CPU smoke 验证（2026-06-10）：Phase 0 drift（真实数据 → CSV/报告/图）+ Phase 1 within EEGNet 训练，均端到端跑通。
- GPU 全量仍走 Slurm + `mi_torch_cu118`，先 smoke 再 full。

运行示例：

```bash
# dry-run 看计划
python code/run.py --dry-run --config code/configs/experiments/phase1_baseline.yaml
# Phase 0 漂移诊断（CPU）
python code/run.py --config code/configs/experiments/phase0_drift_diagnostic.yaml --subjects 1,2
# Phase 1 baseline smoke（GPU 节点）
python code/run.py --config code/configs/experiments/phase1_baseline.yaml \
    --models eegnet --protocol within --subjects 1,2 --folds 2 --max-epochs 3 --device cuda
```

汇总也已迁入 `code/`：`python code/run.py --summarize --config code/configs/experiments/<phase>.yaml` 会生成表/图/原始报告，并额外产出按 9 段结构的 canonical `REPORT.md`（`code/summaries/`）。已用 backup 现成 run CSV 验证 Phase 2b：30150 行、表/图/报告齐全，canonical 报告数字与历史一致。

## 4. SHU 数据集就绪度

可以继续接入开发，但还不能一键复现 WBCIC 全流程。已具备：`code/configs/datasets/shu.yaml`、`code/datasets/shu.py`（只读索引）、`code/datasets/channel_mapping.py`。仍缺：

1. SHU raw/EDF/MAT → 标准 `.npz [trials, 32, 1000]` 的预处理脚本。
2. SHU processed manifest。
3. SHU 版 Phase 0/1/2a/2b 实验配置。
4. 32ch 输入与 WBCIC 58ch 的统一/跨数据集策略。

## 5. 下一步建议（优先级）

1. （已完成）让 `code/run.py` 直连 `code/` 模块，恢复可训练能力。
2. （已完成）迁移 summarize_* 到 `code/summaries/`，并加 canonical 9 段报告。
3. 新增 `code/configs/experiments/phase2c_prototype_drift.yaml` 并实现协议。
4. 补 SHU 预处理 + manifest + SHU 实验配置。
5. 新增 `cross_dataset.yaml`（common / zero_pad 通道策略）。

## 6. 冗余与可删除

| 对象 | 结论 |
|:---|:---|
| `backup/root_archive_2026-06-10/` | 保留，历史可追溯，不要删。 |
| `backup/legacy_snapshot_2026-06-10/` | 与 root_archive 有重叠，确认 root_archive 完整后可删此快照。 |
| 阶段目录里的结果 | 是从 backup 复制来的可读副本，保留。 |
| `/share/home/yuan/SYX/eeg-mi-online` + `/share/home/yuan/SYX/backups` | 明确保留。 |
| `/share/home/yuan/SYX` 其他目录 | 删前先复制到 `/share/home/yuan/SYX/backups/`，不要直接 rm。 |

## 7. 外部数据边界

WBCIC-SHU `/share/workspace2/moto_imagination/WBCIC_SHU` 与 SHU `/share/workspace2/moto_imagination/SHU` 只读，禁止写入/改名/删除。
