---
title: "SHU 2022 Prototype Drift Analysis (Phase 2c) — AI 分析"
tags:
  - "#pipeline/4_analysis"
  - "#modality/eeg"
  - "#method/domain_generalization"
  - "#paradigm/motor_imagery"
created: "2026-07-06"
updated: "2026-07-06"
status: "active"
---

# SHU 2022 Prototype Drift Analysis（Phase 2c）— AI 深度分析

> 诊断实验（非 adaptation）。冻结 source-only baseline，提取 penultimate embedding，计算 class
> prototype 漂移与 cross-session acc_drop 的关系。SHU 25 被试 × 20 有向对 × 3 模型 × 5 seeds =
> **7500 cells 全 ok**（run_status.csv：ok 7500 / missing 0）。metrics 45000 行。数字来自本目录 `tables/`。
>
> **合规声明**：target labels are used only for offline diagnostic analysis, not for training or adaptation
>（`n_target_labels_used_for_training ≡ 0`，`used_target_labels_for_training` 全 False）。

## 0. 核心结论（honest）

1. **SHU 复现了 WBCIC 的机制：跨 session 掉点来自 within-class scatter 膨胀 / Fisher collapse，而非
   centroid collapse。** source→target（label_based/euclidean，n=7500）：类内 scatter **15.70 → 38.27
   (+144%)**、Fisher 比 **1.960 → 0.787 (−60%)**，而类间 separation 反而**增大** 6.60 → 23.94。即表征
   "弥散/糊化"，不是两类原型互相靠拢。
2. **最强的单变量预测子是 `fisher_change`（Spearman ρ=0.43，r²≈0.13）和 `separation_change`
   （cosine ρ=0.38，r²≈0.16）**；`prototype_drift_mean`（ρ=0.18）、`direction_cosine`（ρ=0.05）、
   margin 类都很弱。与 WBCIC 定性一致（Fisher/separation 主导，drift/direction 弱）。
3. **cosine 几何比 euclidean 更线性**：separation_change 在 cosine 下 pearson r=0.40 / r²=0.16，euclidean
   下 pearson≈−0.04 / r²≈0.00。与 WBCIC 结论一致 → 下游 prototype 方法应在 cosine/归一化空间做。
4. **模型依赖同 WBCIC**：EEGNet/DeepConvNet 的 fisher_change 信号清晰（ρ 0.53/0.55），**FBCNet 很弱
   （ρ 0.23）且几何异常**（scatter 几乎不变 29.6→30.4、separation 反而缩小 7.41→5.83）→ FBCNet 掉点是
   另一种成因，不能并入 prototype 结论。
5. **相关强度整体中等偏弱、且比 WBCIC 更噪**：SHU cross 近 chance（acc_target≈0.53），acc_drop 动态范围
   被地板压缩，削弱相关；`prototype_drift_mean` 的 acc_drop 三分位甚至**非单调**（low 0.046 / mid 0.161 /
   high 0.108）。因此 SHU 的 prototype 信号"方向对、机制同、但强度弱、噪声大"。
6. **判定：qualified go（弱化版）**。机制在 SHU 得到复现，支持 Step 4 在 cosine 空间做 Oracle 上限诊断；
   但须以 SHU 低 SNR / 近地板为前提，FBCNet 单独处理。

## 1. 实验目标

验证 SHU 跨 session 掉点是否来自 embedding 空间的 class-prototype 漂移；若成立，为 prototype-based
adaptation（Step 4+）提供理论依据；若不成立，转向其他机制。同时与 WBCIC Phase 2c 做同质性对照。

## 2. 方法定义

- 冻结 source-only baseline（source-train 切 val 早停），提取 source-train / source-val / target-test 的
  penultimate embedding + logits/probs/pred/conf。
- prototype 类型：`label_based` / `confidence_weighted` / `correct_only`；距离：`euclidean` / `cosine`。
- 6 类指标：prototype_drift_mean、prototype_direction_cosine、separation_change、target_(negative_)margin、
  scatter_change、fisher_change；与 acc_drop（= acc_source_val − acc_target）做 Pearson/Spearman/linregress。
- 泄漏断言：source≠target、train/val 不交叠、target label 只离线用。degenerate（correct_only 空类）显式标 status。

## 3. 实验协议

- 20 有向 cross-session 对（5 session）× 25 被试 × 3 模型 × 5 seeds = 7500 cells；每 cell 6 (ptype×dist) 指标行。
- canonical 视角：`label_based` + `euclidean`（并对照 `cosine`），model=ALL（n=7500）。
- 数据入口：SHU `npz_clean` manifest（status=ok）。**注：本轮修正了 summarizer 的 manifest bug**——
  `summarize_from_cfg` 原读 `data.manifest_path`，与全项目约定的 `data.manifest` 不一致，导致 run_status
  误用 WBCIC 期望网格（把 sub-026+ 判为 missing）。修正后 SHU 期望网格正确，7500/7500 全 ok。

## 4. 结果

### 4.1 单变量相关（label_based，model=ALL，n=7500；`prototype_accuracy_correlation.csv`）

| 指标 vs acc_drop | euclidean ρ (Spearman) | euclidean r² | cosine ρ | cosine r² |
|:---|---:|---:|---:|---:|
| `fisher_change` | **0.433** | 0.131 | 0.433 | 0.131 |
| `separation_change` | 0.280 | 0.001 | **0.376** | **0.160** |
| `prototype_drift_mean` | 0.177 | 0.003 | 0.119 | 0.010 |
| `target_margin_mean` | 0.075 | 0.002 | 0.045 | 0.001 |
| `prototype_direction_cosine` | 0.047 | 0.001 | 0.047 | 0.001 |
| `target_negative_margin_rate` | 0.007 | 0.002 | 0.025 | 0.004 |

### 4.2 机制：source → target（label_based/euclidean 均值）

| 范围 | acc_src_val | acc_tgt | acc_drop | separation | scatter | Fisher |
|:---|---:|---:|---:|---:|---:|---:|
| ALL | 0.632 | 0.527 | 0.105 | 6.60 → 23.94 | 15.70 → 38.27 (**+144%**) | 1.960 → 0.787 (**−60%**) |
| eegnet | 0.645 | 0.538 | 0.107 | 4.48 → 36.25 | 7.61 → 47.01 | 2.063 → 0.913 |
| deepconvnet | 0.649 | 0.536 | 0.113 | 7.92 → 29.74 | 9.90 → 37.42 | 2.395 → 0.805 |
| fbcnet | 0.603 | 0.508 | 0.095 | 7.41 → 5.83 | 29.60 → 30.37 | 1.422 → 0.643 |

### 4.3 模型依赖（fisher_change vs acc_drop，label_based/euclidean）

| model | Spearman ρ | r² |
|:---|---:|---:|
| `deepconvnet` | 0.550 | 0.155 |
| `eegnet` | 0.531 | 0.166 |
| `fbcnet` | 0.226 | 0.080 |

## 5. 分析

- **机制复现且更极端**：SHU 的类内 scatter 膨胀（+144%）比 WBCIC（+53%）更剧烈，Fisher collapse（−60%）
  与 WBCIC（−66%）相当。separation 增大而 Fisher 崩塌，说明"类内弥散"压过了"类间拉开"，判别性净损失——
  与"表征糊化"图像一致，排除了 centroid collapse。
- **cosine > euclidean**：separation_change 只有在 cosine 下才线性（r² 0.16 vs 0.00），再次说明 prototype
  的下游几何应归一化。fisher_change 两种距离一致（它由 scatter/separation 比构成，与 prototype 距离无关）。
- **FBCNet 是另一种病**：scatter 几乎不变、separation 缩小、相关弱（ρ 0.23）。FBCNet 的掉点不走"scatter
  膨胀"通道，几何成因不同，须单独建模，不能并入 prototype 结论。
- **弱信号 + 非单调的诚实警示**：drift_mean 的 acc_drop 三分位非单调（mid 最高），且多数单变量 r² < 0.05。
  根因是 SHU cross 近 chance，acc_drop 方差被地板压缩、噪声占比高。SHU 的 prototype 结论"方向可信、强度
  存疑"，不宜照搬 WBCIC 的信心度。

## 6. 与已有阶段的关系

- 承接 SHU Phase 2b：无学习对齐不足（最佳 z-score +1.42pp），负结果指向表征层面重组——Phase 2c 证实了
  scatter/Fisher 机制。
- 对照 WBCIC Phase 2c：机制同质（scatter 膨胀 / Fisher collapse、cosine 更优、FBCNet 弱），量级 SHU 更极端
  但相关更噪。两数据集共同支持"跨 session 掉点 = 类内弥散"的假设。
- 支撑 Step 4：cosine 空间 Oracle 上限诊断 + scatter/reliability 探针。

## 7. 下一步建议

- Step 4（qualified go）：在 **cosine/归一化 embedding** 空间做 target-prototype Oracle 上限诊断，量化"若已知
  target prototype 能回收多少 acc"；并加 scatter/reliability 探针定位可修子集。
- **FBCNet 单独处理**：其几何不符合 scatter-膨胀假设，需单独诊断（可能是 filterbank 特征本身近 chance）。
- 解释 SHU 结果时始终以"near-chance 地板 + 弱信号"为前提；必要时用 WBCIC 做机制锚定、SHU 做稳健性对照。
- 未做/不承诺：任何 adaptation / memory / online / tool-routing 仍为 future，须由 Step 4 结果支撑。

## 8. 文件清单

- `tables/prototype_drift_metrics.csv` — 45000 行 metrics（每 cell × ptype × dist）。
- `tables/prototype_accuracy_correlation.csv` — 6 指标 × ptype × dist × model 的相关（本报告 §4.1 来源）。
- `tables/prototype_table.csv` — 每 class prototype 汇总。
- `tables/run_status.csv` — 7500 cells 完整性（全 ok）。
- `tables/trial_embeddings_index.csv` — trial-level embedding 索引（npz 引用，不含大文件）。
- `figures/` — 各指标 vs acc_drop 散点 + acc_drop_by_model + correlation_summary。
- `report/prototype_drift_report.md`（脚本原生报告）、`report/RUN_STATUS.md`。
- 重型产物：`outputs/experiments/shu/prototype_drift_v1/embeddings/`（npz，约 5.1G，仅索引进表）。
