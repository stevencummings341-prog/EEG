# Step 1 报告 — Multi-source 静态 baseline：ses-01 + ses-02 → ses-03

> **状态：✅ 已完成（2026-06-08）。** 这是 cross-session 静态 baseline 的补齐项（Step 1）。
> 机器生成的简版报告：`outputs/experiments/baseline_v1/provenance/session_multisource_v1/summaries/MULTISOURCE_STEP1_REPORT.md`。
> 本文件是带深入分析（per-subject / drift-level / 上界对比）的完整版。
> 数据入口：`eog_ecg_clean` 的 `status=ok` session；与单源 cross baseline 完全同配方，可直接对比。

---

## 1. 协议（protocol）与无泄漏设计

| 项 | 设定 |
|---|---|
| protocol 名 | `multisource_0102_to_03` |
| train | 同一被试 `ses-01 + ses-02` 的全部 trial 合并（400 trial） |
| val | 只从合并后的 train stratified carve（20% = 80 trial），用于早停 |
| test | 同一被试 `ses-03` 全部 trial（200），**仅用于最终评测** |
| models | EEGNet / DeepConvNet / FBCNet |
| seeds | 0,1,2,3,4（报告 mean ± std） |
| 训练配方 | 与单源 cross baseline 一致：bs=16, lr=1e-3, Adam, max_epochs=100, early-stopping patience=20, val_fraction=0.2 |

**无泄漏（代码层面强制 + 已核验）**：
- `test_session=ses-03` 不在 `train_sessions=[ses-01,ses-02]`（代码断言）。
- train/val 索引互斥，且都索引进"只含 ses-01+02"的合并数组——val 永不含 ses-03。
- `ses-03` 的 label 从不参与训练/验证/早停/调参，只在 `predict` 后算指标。
- 每行结果记录 `n_train=320 / n_val=80 / n_test=200`，可审计。

---

## 2. used / skipped subjects

- **used = 47**（`ses-01/02/03` 全部 ok 的被试）。
- **skipped = 4**（任一 session 非 ok）：

| subject | reason | ok_sessions |
|---|---|---|
| sub-023 | missing ok `ses-01` | ses-02 \| ses-03 |
| sub-024 | missing ok `ses-02, ses-03` | ses-01 |
| sub-026 | missing ok `ses-01` | ses-02 \| ses-03 |
| sub-032 | missing ok `ses-02` | ses-01 \| ses-03 |

结果行总数 = 3 models × 5 seeds × 47 subjects = **705**，其中 ok=705 / failed=0 / NaN=0。

---

## 3. 主结果（mean ± std over 5 seeds，test = ses-03）

| model | Acc | BalAcc | MacroF1 | AUC | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| **EEGNet** | **0.7717 ± 0.003** | 0.7717 | 0.7663 | 0.8258 | 0.5522 | 0.3161 | 0.1020 |
| DeepConvNet | 0.7564 ± 0.007 | 0.7564 | 0.7472 | 0.8175 | 0.6160 | 0.3429 | 0.1179 |
| FBCNet | 0.6750 ± 0.002 | 0.6750 | 0.6488 | 0.7392 | 0.7398 | 0.4401 | 0.1484 |

模型排序 EEGNet > DeepConvNet > FBCNet，与 within / 单源 cross baseline 一致。

---

## 4. 与单源 cross-session 对比（test = ses-03）

| model | ses-01→03 | ses-02→03 | **ses-01+02→03** | Δ vs best **direction** |
|---|---:|---:|---:|---:|
| EEGNet | 0.6991 ± 0.009 | 0.7492 ± 0.008 | **0.7717 ± 0.003** | **+0.0224** |
| DeepConvNet | 0.6757 ± 0.004 | 0.7211 ± 0.009 | **0.7564 ± 0.007** | **+0.0353** |
| FBCNet | 0.6142 ± 0.006 | 0.6484 ± 0.005 | **0.6750 ± 0.002** | **+0.0267** |

> Δ vs best direction = multi-source 总体均值 − 单源中最好的那个方向（这里都是 `ses-02→03`，相邻天最近）。
> 三个模型全部 **正向**，平均 **+0.0281**。**结论：合并两天 source 训练，对 ses-03 泛化稳定有效。**

### 4.1 更严格的对照：vs **per-subject best single**（oracle 选源）

如果允许"为每个被试单独挑它更好的那个单源 session"（一个事后 oracle），multi-source 仍然更高：

| model | per-subject best-single 均值 | multi-source 均值 | Δ | 被试中提升占比 |
|---|---:|---:|---:|---:|
| EEGNet | 0.7577 | 0.7717 | **+0.0139** | 33/47 |
| DeepConvNet | 0.7355 | 0.7564 | **+0.0209** | 33/47 |
| FBCNet | 0.6653 | 0.6750 | **+0.0097** | 27/47 |

即便对手是"逐被试挑最好单源"，multi-source 平均仍占优——说明增益不是简单靠"撞到好的那天"。

### 4.2 与 within-session ses-03 上界对比（漂移代价回收）

| model | 单源最好(ses-02→03) | **multi-source** | within ses-03（上界） | 回收 gap 比例 |
|---|---:|---:|---:|---:|
| EEGNet | 0.7492 | 0.7717 | 0.8228 | ~30%（0.0225 / 0.0736） |
| DeepConvNet | 0.7211 | 0.7564 | 0.7872 | ~53%（0.0353 / 0.0661） |
| FBCNet | 0.6484 | 0.6750 | 0.7359 | ~30%（0.0266 / 0.0875） |

multi-source 把"单源 cross → 同 session 上界"的差距回收了约 **30%–53%**，剩余 gap 留给 Step 2 的 alignment / adaptation。

---

## 5. Per-subject 深入分析（EEGNet 为主）

per-subject 表：`outputs/experiments/baseline_v1/cross_session/tables/cross_by_subject.csv`
（原始 Step 1 版本归档在 `baseline_v1/provenance/session_multisource_v1/summaries/`）。

### 5.1 提升 / 变差分布（Δ vs per-subject best single）

| model | improved | worse | mean Δ | median Δ | min | max |
|---|---:|---:|---:|---:|---:|---:|
| EEGNet | 33/47 | 13 | +0.0139 | +0.0200 | −0.1460 | +0.0940 |
| DeepConvNet | 33/47 | 12 | +0.0209 | +0.0230 | −0.1730 | +0.1790 |
| FBCNet | 27/47 | 18 | +0.0097 | +0.0080 | −0.0980 | +0.1440 |

### 5.2 EEGNet top 提升 / top 变差

| subject | ses-01→03 | ses-02→03 | multi | Δ vs best |
|---|---:|---:|---:|---:|
| sub-047 | 0.776 | 0.743 | 0.870 | **+0.094** |
| sub-040 | 0.725 | 0.668 | 0.807 | **+0.082** |
| sub-021 | 0.708 | 0.713 | 0.789 | **+0.076** |
| sub-003 | 0.828 | 0.765 | 0.903 | **+0.075** |
| sub-022 | 0.566 | 0.740 | 0.807 | **+0.067** |
| … | | | | |
| sub-034 | 0.740 | 0.859 | 0.830 | −0.029 |
| sub-020 | 0.572 | 0.486 | 0.503 | −0.069 |
| sub-029 | 0.441 | 0.935 | 0.831 | −0.104 |
| sub-039 | 0.437 | 0.582 | 0.457 | −0.125 |
| sub-030 | 0.608 | 0.860 | 0.714 | −0.146 |

**变差的规律**：失败案例几乎都是**两个 source session 质量差异极大**（如 sub-029：ses-01=0.441 vs ses-02=0.935；sub-030：ses-02=0.860 但 ses-01 差）。把差的那天混进来会把强 source 往下拉。这直接指向 Step 2：**按 source 质量/漂移加权**、或先做 alignment 再合并，可能比无脑拼接更好。

### 5.3 按 drift_level 分层（EEGNet，Δ vs best single）

| drift_level | n | mean Δ | median Δ |
|---|---:|---:|---:|
| high | 16 | +0.0037 | +0.0180 |
| moderate | 14 | +0.0156 | +0.0055 |
| stable | 17 | +0.0221 | +0.0230 |

stable 被试从合并 source 中获益最稳；high-drift 被试均值增益最小（被极端失败案例拖低），但中位数仍为正——**高漂移被试更需要 Step 2 的 alignment，而不仅是堆数据**。

---

## 6. seed 稳定性

各 seed 的 multi-source 平均 acc（over subjects）：

| model | seed0 | seed1 | seed2 | seed3 | seed4 |
|---|---:|---:|---:|---:|---:|
| EEGNet | 0.7691 | 0.7704 | 0.7682 | 0.7739 | 0.7766 |
| DeepConvNet | 0.7429 | 0.7624 | 0.7595 | 0.7555 | 0.7617 |
| FBCNet | 0.6737 | 0.6727 | 0.6751 | 0.6761 | 0.6776 |

EEGNet / FBCNet 跨 seed 极稳（std ≤ 0.003）；DeepConvNet seed0 略低（0.743），std 0.007，仍可接受。

---

## 7. 无泄漏 / 可靠性核验

- Slurm：train `21240–21244` + summarize `21245`，全部 `COMPLETED`，exit `0:0`。
- 705 rows ok / 0 failed / 0 NaN。
- `n_train=320`、`n_val=80`、`n_test=200`（全被试一致）。
- split JSON（`outputs/experiments/baseline_v1/provenance/session_multisource_v1/splits/`）记录每个 (subject, seed) 的 train/val 索引；train/val 互斥；val 全部来自 ses-01+02。
- `ses-03` label 仅用于最终评测。

---

## 8. 结论

1. **multi-source 静态 baseline 已补齐且有效**：`ses-01+02→ses-03` 对 3 个模型都优于最强单源 `ses-02→03`（+0.022 ~ +0.035），平均 +0.0281。
2. 即便对比"逐被试挑最好单源"的 oracle，multi-source 仍占优（+0.010 ~ +0.021）。
3. 增益回收了"单源 cross → within ses-03 上界"约 30%–53% 的漂移 gap。
4. **失败案例集中在两 source 质量差异大的被试** → 朴素拼接对高漂移被试不是最优，为 Step 2 提供了明确动机。

---

## 9. 下一步（Step 2，不在本轮执行）

- **Step 2 = no-learning adaptation baseline**（不更新模型权重）：`none` / `session_zscore` /
  Euclidean Alignment / Riemannian Alignment / target BN statistics adaptation / filter-bank reweighting。
  细节见 `docs/ADAPTATION_BASELINE_PLAN.md`。
- 可在 Step 2 顺带验证 §5.2 的猜想：对两 source 先 align 再合并、或按漂移加权，是否能救回失败被试。
- online / 41-10 / fine-tuning / CAP-EEGNet full / multi-agent / prototype / memory 仍是 **future（Step 3+）**，未运行。

---

## 10. 文件索引

| 内容 | 路径 |
|---|---|
| 机器版报告 | `outputs/experiments/baseline_v1/provenance/session_multisource_v1/summaries/MULTISOURCE_STEP1_REPORT.md` |
| 原始结果行 | `.../summaries/results_multisource_0102_to_03.csv` |
| per-seed / per-model | `.../summaries/multisource_by_seed.csv`、`multisource_by_model.csv` |
| per-subject 增益 | `outputs/experiments/baseline_v1/cross_session/tables/cross_by_subject.csv` |
| 汇总（schema 对齐） | `.../summaries/summary_by_model_protocol.csv` |
| 图 | `.../figures/multisource_vs_singlesource_acc.png` |
| split JSON | `.../splits/multisource_<subject>_seed<k>.json` |
| checkpoints | `checkpoints/session_multisource_v1/{eegnet,deepconvnet,fbcnet}/` |
| 代码 | `src/evaluation/session_multisource_protocols.py`、`scripts/train_session_multisource.py`、`scripts/summarize_multisource_results.py`、`configs/session_multisource_compare.yaml` |
