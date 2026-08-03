---
title: "Experiment Log"
tags:
  - "#pipeline/4_analysis"
  - "#modality/eeg"
created: "2026-06-10"
updated: "2026-07-06"
status: "active"
---

# Experiment Log

## 2026-07-06 -- SHU Phase 1/2a/2b/2c summarize + AI 分析（done，与 WBCIC 齐平）

补齐 SHU 剩余进度：训练早已完成，本轮做 summarize + AI 分析。完整性：P1 within 18750 + cross 7500 行；
P2a 375 行 ok；P2b 45000 行 ok/0 failed/complete；P2c 45000 metric 行、run_status 7500/7500 ok。
真实结果：P1 within/cross(5-seed) EEGNet 0.611/0.538、DeepConvNet 0.606/0.536、FBCNet 0.553/0.508
（cross 近 chance，地板效应）；P2a ses01+02→03 EEGNet 0.544/DeepConvNet 0.558/FBCNet 0.512（vs 最强单源
+0.7/+2.6/−0.2pp）；P2b 无对齐 0.5274、最佳 session_zscore +1.42pp（≠WBCIC BN-stats）、无方法过 +2pp、
filterbank −1.47pp 有害、high-drift 受益最小；P2c 机制同 WBCIC——scatter 15.7→38.3(+144%)/Fisher
1.96→0.79(−60%)、非 centroid collapse，最强预测子 fisher_change ρ=0.43/separation_change cosine
ρ=0.38，cosine>euclidean，FBCNet 弱且几何异常，信号比 WBCIC 更噪。
两处 bug 修复：P2c summarizer manifest key（data.manifest_path→data.manifest，否则 run_status 误用 WBCIC
网格）；P2b baseline schema（新脚本 scripts/make_baseline_cross_all.py 把 Phase1 cross 转成对齐口径）。
结果落 `2_baseline/shu/{no_alignment_baseline,alignment_baseline}/` 与 `4_experiments/shu/prototype_drift/`，
各写 AI_ANALYSIS.md。target labels 仅离线诊断用（P2c）。SHU Phase 0–2c 全部 done。

## 2026-07-05 -- 重新接手盘面核查：SHU 训练全完成，summarize 待做

重新接手，仅核查未跑新实验。`squeue` 空；SHU 4 phase 训练 CSV 全部齐全且日志无 Traceback/CUDA/RuntimeError：
Phase 1 = 30 CSV（within/cross/meta × 3 models × seeds 0-4，ALL DONE）；Phase 2a = 15 CSV（ses01+02→03）；
Phase 2b = 75 CSV（5 methods × 3 models × seeds 0-4，日志 rows=2500 ok=2500 failed=0）；
Phase 2c = 60 run CSV + 15 metrics CSV。最后一个 CSV 落盘 2026-06-13 05:23。
**关键**：`2_baseline/shu/`、`4_experiments/shu/prototype_drift/` 除 README 外为空，聚合表/canonical REPORT/run_status 均未生成
→ SHU P1/2a/2b/2c 训练 done、**summarize + AI 分析 pending**，正式数字尚不存在（不得引用/编造）。
下一步：按 P1→P2a→P2b→P2c 顺序 `--summarize` 并核验 run_status 后写 AI 分析。

## 2026-06-11 -- SHU Phase 0 跨 session 漂移诊断（done）

SHU 2022 Phase 0 全量跑完（Slurm CPU job 21601，250 within-subject pairs / 25 subjects / 367s，seed 0）。
复用 WBCIC 同一 runner 与指标。核心：MMD 0.356、CSP_sim 0.344、ERD μ/β 0.527/0.532、
RMS ratio median 1.03、fisher_shift≈-0.0012。结论：与 WBCIC 同质（空间+μ/β 频谱漂移主导，
幅值稳定，左右手可分性未塌），但 SHU 空间漂移更重（MMD↑ CSP↓）。分层 high 9 / moderate 8 / stable 8。
结果：`1_session_drift/shu/{report,tables,figures}/` + `report/AI_ANALYSIS.md`；重算源 `outputs/analysis/shu/session_drift_v1/`。
Phase 1 baseline 已提交（GPU job 21602-21604，每 model 一个），running。Phase 2a/2b/2c pending。

## 2026-06-11 -- SHU seed 覆盖核查（确认无重复/覆盖风险）

核查（未 summarize）：21602-4 日志确认仅 seed0；补交 21644-55 命令显式 --seeds N（单 seed），
不会重复跑 0-4。Phase 2a COMPLETED（3 models × seeds 0-4，单 seed/CSV，25 行，无重复）。
Phase 2b（21613 running, seeds=[0], 5 methods, tasks=500）与 2c 均 per model×seed 覆盖 0-4。
Phase 1 runs 目录暂空（seed0 仍在 within，CSV 完成后才落盘）。下一步只等训练结束。

## 2026-06-11 -- SHU 训练全量提交 + Phase 1 seed 修正

SHU Phase 1/2a/2b/2c 训练任务全量提交至 Slurm（mi_torch_cu118，gpu2node/gpu3node）。
首批 Phase 1（21602-4）误用 config 默认 seeds=[0]，仅 seed0；已把 `shu_phase1_baseline.yaml`
改为 seeds=[0,1,2,3,4] 并补提交 seeds 1-4（21644-21655，每 model×seed 一个，per-seed CSV 不覆盖 seed0）。
提交层面 4 个 phase 均覆盖 seeds 0-4：P1=21602-4+21644-55、P2a=21610-12（config 内 5 seeds）、
P2b=21613-27、P2c=21628-42。训练完成前不 summarize、不写 AI 分析、不更新 results.md 正式数值。

## 2026-06-11 -- SHU 2022 接入 + 双数据集并列架构

无新模型实验（数据/工程）。SHU 2022 预处理完成：作者 per-session `.mat` →
`.npz [trials,32,1000]`（仅标签 {1,2}->{0,1} 归一化，不二次预处理），输出
`/share/workspace2/moto_imagination/SHU/processed/npz_clean/`，125 session 全 ok / 25 subjects。
新增 4 个 SHU 实验 config 与数据集无关的 manifest 解耦（`_resolve_manifest`）。结果区/outputs/
checkpoints 改为 `wbci_shu/` 与 `shu/` 并列，每层 README。CPU smoke 验证 SHU phase1/phase2b 跑通。
SHU 全量实验仍 pending（未跑）。

## 2026-06-11 -- Phase 2c Prototype Drift Analysis (submitted / pending)

Implemented and submitted the Phase 2c prototype-drift diagnostic (frozen-model,
source-only training, target test-only; target labels used only for offline
diagnostic analysis). Scope: WBCIC-SHU 2C, status=ok, 3 models (eegnet,
deepconvnet, fbcnet), seeds 0-4, 50 eligible subjects (288 directed cells per
model/seed), 4320 cells total. CPU smoke (subjects 1,2 / eegnet / seed 0 / 3
epochs) passed end-to-end (no leakage, no NaN, all tables/figures/report). Full
run submitted as 15 GPU jobs + 1 CPU summarizer (afterany), Slurm job ids
21536-21551. Status: submitted/pending -- NOT complete until the summarizer runs
and run_status.csv shows all expected cells ok. Results dir:
`4_experiments/prototype_drift/`; heavy artifacts in
`outputs/experiments/prototype_drift_v1/`.

## 2026-06-10 -- P10-style project reorganization

No new experiment was run. The repository was reorganized with P10-style phase folders and a new `code/` framework for multi-dataset experiments. Existing WBCIC-SHU results remain under their original `outputs/` and `checkpoints/` paths to avoid breaking provenance.
