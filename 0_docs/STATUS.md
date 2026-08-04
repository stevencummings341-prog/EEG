---
title: "Status and Readiness"
tags:
  - "#pipeline/5_dl_model"
  - "#modality/eeg"
created: "2026-06-10"
updated: "2026-08-04"
status: "active"
---

# Status and Readiness

> 当前进度 + 能否继续跑 + SHU 就绪度 + 下一步 + 冗余/删除策略。逐条进度记忆在根目录 `progress.md`。

## 1. 一句话现状

WBCIC-SHU 与 SHU 2022 双数据集 Phase 0–2c 均已完成。**Phase 3 已达 pretrained-model-ready 工程态（2026-07-12）**：Round-1 scaffold + mock live-inference 验证 + **WBCIC full A0 complete（4320/4320, max\|Δ\|=0）** + SHU no_tta replay smoke + 交接契约。**真实预训练模型尚未接入**；**formal Oracle / full T3A 未跑**。下一步：学长 checkpoint → 新 adapter → preflight/live smoke → Phase 3B Oracle。路线见 `PHASE3_ROUTE_PLAN.md` + `3_online_adaptation/PHASE3_TTA_DESIGN.md` + `PRETRAINED_MODEL_INTEGRATION_CONTRACT.md`。

## 2. 进度表

| 阶段 | 状态 | 结果位置 |
|:---|:---|:---|
| Phase 0 Drift Diagnostic | done (WBCIC) | `1_session_drift/wbci_shu/` |
| Phase 1 Baseline | done (WBCIC) | `2_baseline/wbci_shu/no_alignment_baseline/` |
| Phase 2a Multi-source | done (WBCIC) | `2_baseline/wbci_shu/no_alignment_baseline/` |
| Phase 2b No-learning Alignment | done (WBCIC) | `2_baseline/wbci_shu/alignment_baseline/` |
| Phase 2c Prototype Drift | done (WBCIC，4320 cells 全 ok；AI 分析已写) | `4_experiments/wbci_shu/prototype_drift/`；heavy: `outputs/experiments/wbci_shu/prototype_drift_v1/` |
| SHU Phase 0 Drift | done (SHU，250 pairs/25 subj；AI 分析已写) | `1_session_drift/shu/` |
| SHU Phase 1 Baseline | **done**（within/cross 5-seed；AI 分析已写） | `2_baseline/shu/no_alignment_baseline/` |
| SHU Phase 2a Multi-source | **done**（ses01+02→03；并入 no_alignment_baseline） | `2_baseline/shu/no_alignment_baseline/` |
| SHU Phase 2b Alignment | **done**（45000 行全 ok；AI 分析已写） | `2_baseline/shu/alignment_baseline/` |
| SHU Phase 2c Prototype Drift | **done**（7500 cells 全 ok；AI 分析已写） | `4_experiments/shu/prototype_drift/` |
| **Phase 3 TTA backend + pretrained readiness** | **engineering ready to receive real pretrained model**（非 full T3A；非 formal Oracle） | 代码 `code/tta/` + `session_tta.py`；契约 `3_online_adaptation/PRETRAINED_MODEL_INTEGRATION_CONTRACT.md`；结果 `4_experiments/{wbci_shu,shu}/tta/`（含 full A0 + SHU smoke） |
| Phase 4+ Memory/Online/Agent | future | `3_online_adaptation/` |

> **Phase 3 工程验收（2026-07-12）**：mock live path validated；WBCIC full A0 complete；SHU replay smoke passed；tests/tta 41 passed。未接入真实预训练模型。

Phase 2c full run: 50 eligible subjects (47×6 + 3×2 = 288 cells/(model,seed)), 3 models, 5 seeds = 4320 cells; smoke (subjects 1,2, eegnet, seed 0, 3 epochs, CPU) 已端到端通过（无泄漏、无 NaN、表/图/报告齐全），smoke 产物隔离在 `outputs/experiments/prototype_drift_v1_smoke/`。检查命令见 `outputs/experiments/prototype_drift_v1/RUN_PLAN.md`。

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

## 4. SHU 数据集就绪度（2026-07-06：Phase 0–2c 全部完成）

> 更新：SHU Phase 0/1/2a/2b/2c 已全部训练 + summarize + AI 分析完成（见 §2 表与 progress 2026-07-06）。下面的"可直接跑"说明保留为复跑/重训参考。

数据与配置全部就绪，可直接对 SHU 开跑：

1. ✅ 预处理脚本 `scripts/preprocess_shu.py`（核心 `code/preprocessing/shu_mat.py`）：作者 per-session `.mat` → `.npz [trials,32,1000]`，仅做标签 {1,2}->{0,1} 归一化。
2. ✅ SHU processed manifest 已生成：`/share/workspace2/moto_imagination/SHU/processed/npz_clean/processed_manifest.csv`（125 session 全 ok / 25 subjects）。
3. ✅ SHU 实验配置：`code/configs/experiments/shu_phase{0_drift_diagnostic,1_baseline,2b_alignment,2c_prototype_drift}.yaml`。
4. 通道策略：SHU 用原生 32ch 独立跑（各数据集各自基准，不混合）；跨数据集 common/zero_pad 策略仍为 future。

SHU 运行示例（GPU 节点）：

```bash
python code/run.py --config code/configs/experiments/shu_phase1_baseline.yaml --device cuda
python code/run.py --config code/configs/experiments/shu_phase2b_alignment.yaml --device cuda
python code/run.py --config code/configs/experiments/shu_phase2c_prototype_drift.yaml --device cuda
```

已用 CPU smoke 验证 phase1（within 10 行 + cross 20 对）与 phase2b（EA/RA 32ch 40 行全 ok）端到端跑通。

## 5. 下一步建议（优先级，2026-07-12 — pretrained-model-ready）

> 工程验收已完成（A0 + mock live + SHU smoke + 契约）。权威路线仍见
> `PHASE3_ROUTE_PLAN.md` / `PHASE3_TTA_DESIGN.md` / `PRETRAINED_MODEL_INTEGRATION_CONTRACT.md`。

1. **（当前）等学长预训练模型**：按契约交付物新增 `code/tta/adapters/<name>.py` + config，**不重写** TTA backend。
2. **接入后**：preflight → live inference smoke →（可选）与 Phase 2c baseline 对照 → minimal T3A。
3. **Phase 3B Oracle 裁决门**（收益+风险双条件；阈值 provisional）——通过才扩大 T3A。
4. **禁止提前做**：full T3A sweep / 大规模 ablation / Tent/SHOT（除非裁决后明确需要）。
5. **FBCNet / SHU 单列**，不并入 WBCIC×{EEGNet,DeepConvNet} 主结论。
6. （future）safe-T3A / prototype memory / online / EEG foundation models。

## 6. 冗余与可删除

| 对象 | 结论 |
|:---|:---|
| `backup/root_archive_2026-06-10/` | 保留，历史可追溯，不要删。 |
| `backup/legacy_snapshot_2026-06-10/` | 与 root_archive 有重叠，确认 root_archive 完整后可删此快照。 |
| 阶段目录里的结果 | 是从 backup 复制来的可读副本，保留。 |
| 本项目克隆目录 + 本机备份目录 | 明确保留。 |
| 本机账号下其他无关目录 | 删前先备份，不要直接 rm。 |

## 7. 外部数据边界

外部路径由本机 `code/configs/paths.local.yaml` / `datasets/*.local.yaml` 配置（仓库内为 `/CHANGE/ME` 占位，见 `SETUP.md`）。**raw 只读**；唯一允许写外部盘的是 local 配置显式指向的 `processed/` 子树。换机不要把填好的 `*.local.yaml` 推到 GitHub。
