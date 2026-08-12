---
title: "Status and Readiness"
tags:
  - "#pipeline/5_dl_model"
  - "#modality/eeg"
created: "2026-06-10"
updated: "2026-08-10"
status: "active"
---

# Status and Readiness

> 当前进度 + 能否继续跑 + SHU 就绪度 + 下一步 + 冗余/删除策略。逐条进度记忆在根目录 `progress.md`。

## 1. 一句话现状

**2026-08-10：WBCIC 3C 论文对齐 LOSO 已完成**——5 foundation（`foundation_3c_loso_paper_v1`）+
ATCNet arm A（同 recipe）+ arm B（论文 recipe，`atcnet_3c_loso_paper_recipe_v1`）全部 11/11 ok。
对标 DSGNet Acc **0.6856** / ATCNet **0.6834**。实测：ATCNet(A) **0.7129**、ATCNet(B) **0.6891**、
最好 foundation flatten **0.6599**。差距分解见 `DSGNET_SHUv5_3C_ANCHOR.md`。旧 run
`foundation_3c_loso_v1`（subject-val）不作对比。DSGNet 本仓复现仍 deferred。
路线见 `FOUNDATION_E2E_ROUTE_PLAN.md`；论文 PDF `inbox/papers/dsgnet_jbhi2026_FullText.pdf`。

**2026-08-10 新增，运行中：统一对比 run `paper_baseline_3c_821_v1`（四卡 36928-36931）。**
7 个模型同一 run / 同划分 / 同配方：4 个已发表 baseline（EEGNet [18] / EEGNeX [20] /
EEG-Deformer [23] / ATCNet [24]，**只用各自作者官方代码**，出处与偏差见
`code/models/paper_baselines/README.md`）+ 我们 3 个（flatten / s4erp / transformer）。
划分 = **8:2:1 跨被试**；配方 = **论文 recipe**（Adam 1e-4 / batch 128 / max 500ep）+ 早停
patience 100；新增 **per-epoch train/val/test 三曲线**（test 仅监控，模型选择只看 val）。
两个 67M 模型用**梯度累积**保持等效 batch 128（`train.micro_batch_per_model`）。
EEG-Inception [27]、MDGEEG [35]、EEG-DG [38]、DSGNet 因无完整官方代码排除。
汇总/绘图：`scripts/summarize_cross_subject.py`、`scripts/plot_three_curves.py`。

Phase 0–2c（双数据集）全部完成，结果保留只读。**Phase 3（跨 session 修复 / TTA）= paused 不是废弃**：
`code/tta/` + `4_experiments/*/tta/` + `PHASE3_ROUTE_PLAN.md` 全部保留，工程态仍是
pretrained-model-ready（WBCIC full A0 complete 4320/4320 max\|Δ\|=0；formal Oracle / full T3A 未跑）。

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
| **Phase 3 TTA backend + pretrained readiness** | **paused（不是废弃）**；工程上 ready to receive real pretrained model（非 full T3A；非 formal Oracle） | 代码 `code/tta/` + `session_tta.py`；契约 `3_online_adaptation/PRETRAINED_MODEL_INTEGRATION_CONTRACT.md`；结果 `4_experiments/{wbci_shu,shu}/tta/`（含 full A0 + SHU smoke） |
| **E2E 端到端基础模型 × 跨被试（当前主线）** | **WBCIC 3C 论文对齐 LOSO done（2026-08-10）**：5 foundation + ATCNet A/B 全 ok；DSGNet 复现 deferred | `outputs/.../foundation_3c_loso_paper_v1/` + `.../atcnet_3c_loso_paper_recipe_v1/`；锚点 `4_experiments/wbci_shu/foundation_cross_subject/DSGNET_SHUv5_3C_ANCHOR.md` |
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

**端到端主线（2026-08-04 新增，已可运行；`--summarize` 尚未支持）**：

```bash
# 极小 CPU smoke（输出隔离到 *_smoke，不污染 v1）
python code/run.py --config code/configs/experiments/shu_foundation_cross_subject.yaml \
    --models s4erp --folds 3 --folds-subset 0 --max-subjects 6 --max-epochs 2 \
    --batch-size 32 --num-workers 0 --device cpu \
    --out outputs/experiments/shu/foundation_cross_subject_smoke \
    --ckpt-dir checkpoints/shu/foundation_cross_subject_smoke
# GPU 全量（Slurm）；重复执行同一条命令 = 断点续跑
python code/run.py --config code/configs/experiments/foundation_cross_subject.yaml --device cuda
```

注意：CPU 上 `s4erp` 单 step 约 10 s（1000 点 × 32ch，ShallowNet 空间卷积是瓶颈），所以
**CPU 只适合 2 epoch 级别的 smoke，正式 smoke 也要上 GPU 节点**。

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

## 5. 下一步建议（优先级，2026-08-04 — 端到端主线）

> 权威路线 = 根目录 `FOUNDATION_E2E_ROUTE_PLAN.md`；协议讨论 = `4_experiments/CROSS_SUBJECT_PROTOCOL_MEMO.md`；
> 文献依据 = `inbox/cross_subject_protocol_research.md`。

1. **（当前，需用户执行）把协议备忘发学长确认 7 个问题**：LOSO vs 5-fold、是否严格零样本、
   session 池化与否、是否加 Euclidean Alignment arm、epoch 预算、先跑哪几个模型、SHU 地板效应怎么写。
2. **确认后**：改两个 config 的 `cross_subject` / `train` 段 → GPU 节点 smoke（1 fold × `s4erp`，
   实测单 epoch 耗时与显存）→ 定 epoch/batch → 全量（Slurm，可断点续跑）。
3. **禁止提前做**：在协议确认前把任何一次 full run 的数字当结果写进论文/汇报；两个数据集合并训练；
   为了省事把 `dualcd_*` 的 `d_model` 改小（会和学长的参数量表对不上）。
4. **报告口径**：per-subject 表格 + mean ± std over subjects，`best.pt` 与 `last.pt` 两套数字并排。
   SHU 要同时标注 chance band（51.4–53.7%），近 chance 时 1–2pp 差异按噪声处理。
5. **Phase 3（paused）**：等端到端 backbone 出来后再回接 Oracle/T3A；`code/tta/` 不动、不重写。
6. （future）prototype memory / online test-then-update / agent。

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
