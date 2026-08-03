---
title: "SHU 2022 No-alignment Baseline (Phase 1 + 2a) — AI 分析"
tags:
  - "#pipeline/4_analysis"
  - "#modality/eeg"
  - "#method/domain_generalization"
  - "#paradigm/motor_imagery"
created: "2026-07-06"
updated: "2026-07-06"
status: "active"
---

# SHU 2022 无对齐 Baseline（Phase 1 + Phase 2a）— AI 深度分析

> 数据集：SHU 2022（25 被试 × 5 session × 32ch）。协议：within-session 10-fold CV + single-source
> cross-session（20 有向对）+ multi-source（ses-01+02 → ses-03）。模型：EEGNet / DeepConvNet /
> FBCNet，5 seeds（0-4）。所有数字来自本目录 `tables/`，未人工修改。

## 0. 核心结论（honest）

1. **SHU 跨 session 掉点存在但绝对幅度小，因为解码本身就接近随机下限**。within-session 三模型仅
   0.55–0.61，single-source cross 0.51–0.54（接近 0.5）。掉点 EEGNet 7.3pp / DeepConvNet 7.0pp /
   FBCNet 4.5pp，比 WBCIC（EEGNet 9.6pp、绝对水平 0.81→0.71）小，但这是**地板效应**：cross 已贴近
   chance，可下降空间被压缩，不代表 SHU 更"稳"。
2. **模型排序与 WBCIC 一致**：EEGNet ≈ DeepConvNet > FBCNet，within 与 cross 都成立。FBCNet 在 SHU
   32ch 上尤其弱（cross 0.508，几乎 chance）。
3. **Multi-source（ses-01+02→ses-03）给出小幅正收益**：DeepConvNet +2.6pp、EEGNet +0.7pp、
   FBCNet −0.2pp（对最强单源）。方向与 WBCIC 一致（多源≥最强单源），但幅度小、FBCNet 甚至轻微反向。
4. **诚实边界**：本阶段 SHU 训练已全部完成、summarize 已生成；但 SHU 解码接近 chance 意味着后续
   prototype/adaptation 分析要以"低 SNR / 近地板"为前提解释，不能照搬 WBCIC 的信心。

> ⚠️ **脚本原生报告的已知不适用项**：`SESSION_MODEL_COMPARE_REPORT.md` 与
> `REPORT_phase1_baseline.md` 内嵌的是 **WBCIC 专用参照常量**（论文 EEGNet 85.32 / DeepConvNet 84.47 /
> FBCNet 78.40、"148 sessions"、"288 pairs"、ses-01/02/03 session-trend）。这些对 SHU **不成立**
> （SHU=25 被试 / 125 sessions / 每 model-seed 500 有向对 / 5 session）。原生报告因此把 SHU 标为
> "INCOMPLETE"，实为 **false alarm**：SHU 数据是完整的（3 models × 5 seeds 全齐，within 18750 行、
> cross 7500 行/含全部 seeds）。以本 AI 分析与 `tables/` 的真实数字为准。

## 1. 实验目标

在 SHU 2022 上复现 WBCIC 的跨 session baseline，量化同被试跨天解码的性能下降，作为 SHU 侧对齐
（Phase 2b）与 prototype drift（Phase 2c）的对照基准；并检验多源训练能否回收部分 gap。

## 2. 方法定义

- **within-session**：每个 ok session 内 StratifiedKFold（10 折），val 仅从 train 切出。
- **single-source cross-session**：同被试有向对 train ses-i → test ses-j（i≠j，均 ok），5 session = 20 对。
- **multi-source**：train = ses-01 + ses-02 全部 trial → test = ses-03（仅最终评估）。
- 统一 trainer（CE + 早停 patience 20）、统一指标（acc/bacc/macro-F1/AUC/NLL/Brier/ECE），5 seeds 取 mean±std。

## 3. 实验协议

- 数据入口：SHU processed `npz_clean` manifest（status=ok，125 session / 25 subjects），32ch × 1000 samples @ 250Hz。
- 无泄漏：test session 的 label 绝不进入 train/val/早停；multi-source 的 val 只从 ses-01+02 切。
- 硬件/环境：GPU 训练（Slurm，`mi_torch_cu118`），summarize CPU（`mi_torch`）。seeds 0-4。

## 4. 结果

### 4.1 Within vs cross（mean±std across 5 seeds，`summary_by_model_protocol.csv`）

| model | within Acc | cross Acc | drop (pp) | within AUC | cross AUC |
|:---|---:|---:|---:|---:|---:|
| `eegnet` | 0.611±0.004 | 0.538±0.005 | 7.3 | 0.638 | 0.553 |
| `deepconvnet` | 0.606±0.004 | 0.536±0.003 | 7.0 | 0.637 | 0.563 |
| `fbcnet` | 0.553±0.006 | 0.508±0.001 | 4.5 | 0.572 | 0.521 |

### 4.2 Multi-source ses-01+02 → ses-03（`multisource_by_model.csv` + `cross_by_direction.csv`）

| model | ses-01→03 | ses-02→03 | **ses-01+02→03** | Δ vs best single |
|:---|---:|---:|---:|---:|
| `eegnet` | 0.537±0.011 | 0.532±0.007 | **0.544±0.013** | +0.7pp |
| `deepconvnet` | 0.532±0.007 | 0.531±0.009 | **0.558±0.012** | +2.6pp |
| `fbcnet` | 0.514±0.007 | 0.503±0.005 | **0.512±0.007** | −0.2pp |

## 5. 分析

- **地板效应是主线索**：SHU cross-session AUC 0.52–0.56、acc 0.51–0.54，逼近随机。掉点 pp 小是因为
  cross 触及地板，不是漂移小。Phase 0 已显示 SHU 空间漂移比 WBCIC 更重（MMD 0.356>0.238），与"低跨天
  可解码性"自洽。
- **FBCNet 在 SHU 尤其失效**：cross 0.508、macro-F1 0.420，几乎不分类。这与 WBCIC 中 FBCNet 偏弱一致，
  但在 32ch SHU 上更极端，提示 FBCNet 的 filterbank/空间先验不匹配 SHU。
- **Multi-source 收益随模型不同**：DeepConvNet 得益最多（+2.6pp），说明合并两源确能增加对 ses-03 的
  泛化；FBCNet 反向（−0.2pp）表明当单源本身近 chance 时，多源无从"投票"。
- **校准**：cross 的 NLL 明显升高（如 eegnet within 0.669 → cross-direction 多在 1.0–1.5），说明跨 session
  不仅 acc 掉，置信度也失准——为 Phase 2c 的 margin/scatter 分析埋下动机。

## 6. 与已有阶段的关系

- 承接 SHU **Phase 0**：空间+μ/β 频谱漂移（MMD 0.356、CSP_sim 0.344）解释了为何 cross 近 chance。
- 对照 **WBCIC baseline**：排序一致、多源有效方向一致；但 SHU 绝对水平低一档（WBCIC EEGNet 0.807/0.711
  vs SHU 0.611/0.538），是"更难、更近地板"的对照数据集。
- 支撑 SHU **Phase 2b**：cross 残余 gap 是否能被无学习对齐回收。
- 支撑 SHU **Phase 2c**：近 chance + 高 NLL 指向 embedding/prototype 层面的可分性退化。

## 7. 下一步建议

- 已完成：Phase 2b alignment、Phase 2c prototype drift 的 summarize + AI 分析（见对应目录）。
- SHU 的低 SNR 意味着：后续任何 adaptation 的"上限"都受限于 within 本身只有 ~0.6；解释 prototype 结果
  时须以此为基线，避免把地板噪声当作机制信号。

## 8. 文件清单

- `tables/summary_by_model_protocol.csv` — within/cross per-model mean±std（主表）。
- `tables/cross_by_direction.csv` — 20 有向对 per-model 准确率。
- `tables/multisource_by_model.csv` — ses-01+02→ses-03 结果。
- `tables/results_within_session.csv` / `results_cross_session.csv` — 原始行。
- `figures/` — within boxplot、cross 矩阵、protocol 对比、multi-vs-single。
- `report/SESSION_MODEL_COMPARE_REPORT.md`、`MULTISOURCE_STEP1_REPORT.md`、`REPORT_phase1_baseline.md`、
  `REPORT_phase2a_multisource.md` — 脚本原生报告（注意 §0 的 WBCIC 参照常量不适用于 SHU）。
