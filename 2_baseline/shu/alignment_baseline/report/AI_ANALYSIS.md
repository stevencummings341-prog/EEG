---
title: "SHU 2022 No-learning Alignment Baseline (Phase 2b) — AI 分析"
tags:
  - "#pipeline/4_analysis"
  - "#modality/eeg"
  - "#method/domain_generalization"
  - "#paradigm/motor_imagery"
created: "2026-07-06"
updated: "2026-07-06"
status: "active"
---

# SHU 2022 无学习对齐 Baseline（Phase 2b）— AI 深度分析

> 数据集：SHU 2022（25 被试 × 5 session × 32ch）。协议：single-source 有向对（20 对），5 种无学习/
> 无监督 test-time 对齐 vs 无对齐参照。模型 EEGNet/DeepConvNet/FBCNet，5 seeds。所有数字来自本目录
> `tables/`。总行数 45000（对齐 37500 + none_reference 7500），ok 45000，failed 0，NaN 0，complete=True。

## 0. 核心结论（honest）

1. **无学习统计对齐在 SHU 上同样不足**：没有任何方法达到预登记的 +2pp（≥+0.02）成功线。
   无对齐参照 cross-acc = **0.5274**。
2. **SHU 的最佳方法是 `session_zscore`（+1.42pp → 0.5416），不是 BN-stats**（这点与 WBCIC 不同，
   WBCIC 最佳是 BN-stats +0.77pp）。SHU 上 5 个方法里 **4 个净正向**，1 个（filterbank）明显有害。
3. **`filterbank_reweighting` 在 SHU 显著有害（−1.47pp → 0.5127）**，且在所有漂移等级都为负——与 WBCIC
   中 filterbank 在 high-drift 下暴跌的观察方向一致，在 SHU 32ch 上则是全域有害。
4. **漂移等级交互与 WBCIC 同向**：收益随稳定性递增（session_zscore：stable +2.88pp、moderate +1.06pp、
   high +0.44pp）。**最需要帮助的 high-drift 被试，受益最小**。
5. 结论与 WBCIC 一致：纯统计对齐无法闭合跨 session gap，客观支持（但本阶段不实现）学习型 target 适配。

## 1. 实验目标

检验无监督、纯统计的 test-time 对齐（不使用 target label、不在 target 上学习权重）能否回收 SHU 跨
session 的准确率下降（Phase 1 测得 single-source ≈ 4.5–7.3pp drop、且近 chance）。

## 2. 方法定义

- `none_reference` — 无对齐，来自 Phase 1 single-source cross 行（经 schema 适配为对齐口径，见 §3）。
- `session_zscore` — 逐通道 mean/std 归一化。
- `euclidean_alignment` — 以 R^{-1/2} 白化，R = trial 协方差算术均值。
- `riemannian_alignment` — 以 G^{-1/2} 白化，G = log-Euclidean SPD 均值。
- `bn_statistics_adaptation` — source 训练后，用无标签 target X 刷新 BatchNorm running stats（无 optimizer.step）。
- `filterbank_reweighting` — θ/μ/β/low-γ 子带重加权，使目标频带能量对齐 source。
- 对齐统计量只用 source train 或 target 的无标签 X；`y_test` 只用于最终评估。

## 3. 实验协议

- single-source 有向对：ses-i → ses-j（i≠j 均 ok），每被试 20 对；5 methods × 3 models × 5 seeds × 500 对。
- 无泄漏核查（脚本 RUN_STATUS）：`used_target_y_for_training == False` 全部成立；对齐方法均只用 target 无标签 X。
- **baseline schema 适配**：Phase 1 cross 产出用 `accuracy/train_session`，而对齐汇总的 none_reference
  join 需要 `acc/train_sessions/training_scope`。用 `scripts/make_baseline_cross_all.py` 把 Phase 1 cross
  行转成对齐口径（7500 行，training_scope=single_source），写到 config `baseline_cross_all` 指向的路径。
  join key = (model, train_sessions, test_session, subject, seed)，行对行匹配。

## 4. 结果

### 4.1 各方法平均准确率与相对无对齐的增益（`alignment_by_method.csv` + `alignment_vs_baseline.csv`，all scope）

| method | mean Acc | Δacc vs none | 判定 |
|:---|---:|---:|:---|
| `session_zscore` | 0.5416 | **+0.0142** | 最佳，仍 < +0.02 |
| `riemannian_alignment` | 0.5348 | +0.0074 | 净正 |
| `bn_statistics_adaptation` | 0.5330 | +0.0056 | 净正 |
| `euclidean_alignment` | 0.5318 | +0.0044 | 净正 |
| `none_reference` | 0.5274 | 0（参照） | — |
| `filterbank_reweighting` | 0.5127 | **−0.0147** | 有害 |

### 4.2 增益 × 漂移等级（`alignment_gain_by_drift_level.csv`，Δacc vs none）

| method | stable | moderate | high |
|:---|---:|---:|---:|
| `session_zscore` | +0.0288 | +0.0106 | +0.0044 |
| `riemannian_alignment` | +0.0122 | +0.0063 | +0.0042 |
| `bn_statistics_adaptation` | +0.0098 | +0.0027 | +0.0044 |
| `euclidean_alignment` | +0.0087 | +0.0026 | +0.0022 |
| `filterbank_reweighting` | −0.0166 | −0.0146 | −0.0131 |

## 5. 分析

- **没有方法过 +2pp**：即便最佳的 z-score 也只有 +1.42pp；在 SHU 近 chance 的地板上，这点提升几乎没有
  实用意义。核心 negative 结论与 WBCIC 一致且更强。
- **最佳方法换人（z-score > Riemannian > BN-stats）**：SHU 32ch 上简单的逐通道 z-score 反而最有效，说明
  SHU 的跨天差异里"通道尺度/偏置漂移"占比相对更大；而 WBCIC 上 BN-stats 最优。协方差白化（EA）收益最小。
- **filterbank 全域有害**：把目标频带能量强行拉回 source 频谱，在 SHU 上破坏了本就微弱的判别信息——呼应
  Phase 0 里 SHU 频谱漂移（ERD μ/β 0.53）较重，硬对齐频带会抹掉类信息。
- **high-drift 被试最不受益**：所有方法的收益都 stable > moderate > high，最需要修的被试恰恰最难修——这正是
  "需要学习型适配"的直接证据。

## 6. 与已有阶段的关系

- 承接 SHU Phase 1/2a：cross 残余 gap（近 chance）无法被无学习对齐闭合。
- 对照 WBCIC Phase 2b：核心 negative 结论一致（无方法过 +2pp、high-drift 最难帮）；差异是 SHU 最佳方法
  是 z-score 而非 BN-stats，且 filterbank 在 SHU 全域有害。
- 支撑 SHU Phase 2c：负结果把矛头指向 task representation / prototype drift 假设。

## 7. 下一步建议

- 若要给 SHU 一个便宜的默认前端，用 `session_zscore`（最稳、正向、从不明显有害），而非 WBCIC 的 BN-stats。
- 学习型 Step-3 适配（online test-then-update / adapter / prototype-memory）应聚焦 high-drift 被试——这是
  无学习对齐失败最明显之处。本阶段不实现。

## 8. 文件清单

- `tables/alignment_by_method.csv` — 各方法 mean±std（含 none_reference）。
- `tables/alignment_vs_baseline.csv` — 各方法相对无对齐的 Δacc。
- `tables/alignment_gain_by_drift_level.csv` — 增益 × 漂移等级。
- `tables/alignment_by_direction.csv` / `alignment_by_subject.csv` / `results_alignment_all.csv` — 明细。
- `tables/run_status.csv` — 完整性（45000 行全 ok）。
- `figures/` — method 对比、vs-baseline gain、drift-level、by-subject、by-direction。
- `report/ALIGNMENT_BASELINE_REPORT.md`（13 节原生报告）、`REPORT_phase2b_alignment.md`（canonical）、`RUN_STATUS.md`。
