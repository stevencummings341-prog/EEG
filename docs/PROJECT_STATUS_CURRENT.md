# PROJECT STATUS — CURRENT（权威状态页 / single source of truth）

> 读这一份就能对齐"现在做到哪、下一步做什么"。更细：逐日 `docs/PROGRESS.md`；
> 全量总览 `docs/PROJECT_OVERVIEW.md`；结果 `docs/RESULTS_SUMMARY.md`；
> Step 1 深度报告 `docs/MULTISOURCE_STEP1_REPORT.md`；下一步计划 `docs/NEXT_EXPERIMENT_PLAN.md` +
> `docs/ADAPTATION_BASELINE_PLAN.md`；P10 整合 `docs/P10_INTEGRATION_SUMMARY.md`。
>
> 最后更新：2026-06-08（Step 1 完成）；2026-06-09 文档系统性修复 + **Step 2 alignment baseline
> 已完成（negative/diagnostic 结果：无学习对齐不足）**。

---

## 0. 一句话现状

项目主线 = **EEG MI 跨 session 域泛化（cross-session domain generalization）**。
已完成：漂移诊断 → 静态 baseline（within / 单源 cross / **多源 cross**）→ **Step 2 no-learning
adaptation baseline（已完成）**。Step 2 结论是 **negative/diagnostic：无学习统计对齐不足**——没有任何方法
达到 +2pp；none_reference acc 0.6818，最好 bn_statistics_adaptation 0.6889（Δ +0.0071），EA/RA 反而略降，
z-score/filter-bank 近中性，high-drift 被试受益最小 → 客观支持后续学习型适配（Step 3，**未运行**）。
online / 41-10 / fine-tuning / CAP-EEGNet full / multi-agent / prototype / memory 均为 future，未运行。

---

## 1. 主线进度

| Step | 内容 | 状态 |
|---|---|---|
| A | Session drift 诊断（144 pairs / 50 subjects，9 指标） | ✅ 已完成 |
| B | 静态 baseline：within-session 10-fold CV + 单源有向 cross-session（EEGNet/DeepConvNet/FBCNet，5 seeds，26 520 trainings，30/30 cells，无泄漏/NaN） | ✅ 已完成 |
| Step 1 / C | 静态 baseline 补齐：**multi-source `ses-01+ses-02 → ses-03`**（47 被试，4 跳过，705 rows） | ✅ 已完成 |
| Step 2 / D | **no-learning adaptation baseline**（none_reference / session_zscore / Euclidean / Riemannian(log-Euclidean) / target BN-stats / filter-bank reweighting） | ✅ 已完成（30,150 rows，0 failed/0 NaN）。**结论：无学习对齐不足**（best BN-stats Δ+0.0071，无方法过 +2pp）|
| Step 3+ | online / adapter / prototype / memory / CAP-EEGNet full / 41-10 / fine-tuning / LOSO | 🚧 future，未运行/未验证（Step 2 negative 结果为其提供依据）|

---

## 2. 关键结果（已完成）

### 2.1 漂移诊断（A）
MMD 0.238、CSP 相似度 0.420、ERD-μ corr 0.419、ERD-β 0.482、μ-KS 0.246、RMS 中位数 0.992、Fisher≈0
→ 漂移主要是**空间模式 + μ/β 频谱**，幅值稳定。

### 2.2 静态 baseline（5 seed，mean ± std）

| 模型 | within CV | 单源 cross | drop | **多源 ses-01+02→ses-03** |
|---|---:|---:|---:|---:|
| EEGNet | 0.807 | 0.711 | 11.9% | **0.7717 ± 0.003** |
| DeepConvNet | 0.766 | 0.681 | 11.1% | **0.7564 ± 0.007** |
| FBCNet | 0.720 | 0.628 | 12.8% | **0.6750 ± 0.002** |

### 2.3 Step 1 multi-source 对比单源（test = ses-03）

| 模型 | ses-01→03 | ses-02→03 | ses-01+02→03 | Δ vs best 单源方向 |
|---|---:|---:|---:|---:|
| EEGNet | 0.6991±0.009 | 0.7492±0.008 | **0.7717±0.003** | +0.0224 |
| DeepConvNet | 0.6757±0.004 | 0.7211±0.009 | **0.7564±0.007** | +0.0353 |
| FBCNet | 0.6142±0.006 | 0.6484±0.005 | **0.6750±0.002** | +0.0267 |

- 三个模型 multi-source 全部优于最强单源（平均 +0.0281）；即便 vs 逐被试 oracle 单源仍占优。
- 回收"单源 cross → within ses-03 上界"约 30%–53% 的漂移 gap。
- 失败案例集中在两 source 质量差异大的被试（如 sub-029/sub-030）→ Step 2 alignment 的动机。
- 详见 `docs/MULTISOURCE_STEP1_REPORT.md`。

数据/报告路径：
- **整合看这里**：`docs/RESULTS_SUMMARY.md`（文字版整合报告）+
  `outputs/experiments/baseline_v1/`（`figures/` 全部 21 张图集 + `BASELINE_REPORT.md` 带图报告）
- `outputs/experiments/baseline_v1/provenance/session_model_compare_v1/summaries/`（within + single-source cross 原始）
- `outputs/experiments/baseline_v1/provenance/session_multisource_v1/summaries/`（Step 1 原始 + `multisource_by_subject.csv`）
- `outputs/analysis/session_drift_v1/`（drift 原始）

---

## 3. Step 2（已完成 — negative/diagnostic 结果）

no-learning adaptation baseline（不更新模型权重）：`none_reference`（取自 baseline_v1，不重跑）/
`session_zscore` / Euclidean Alignment / Riemannian Alignment（log-Euclidean SPD mean，numpy/scipy
自实现，不依赖 pyriemann）/ target BN statistics adaptation / filter-bank reweighting。
模型 EEGNet/DeepConvNet/FBCNet，seeds 0-4，协议 = 单源 6 有向对（288）+ 多源 `ses-01+02→ses-03`（47）。
75 GPU jobs `21261-21335` + summarizer `21336` 全部 COMPLETED；30,150 rows，0 failed / 0 NaN。

**结论（诚实）**：无学习统计对齐**不足**，无方法达到 +2pp 成功线。
- none_reference acc **0.6818**；best `bn_statistics_adaptation` **0.6889**（Δ **+0.0071**，唯一正向但远不足）。
- `filterbank_reweighting` −0.0030、`session_zscore` −0.0038（近中性）；`riemannian_alignment` −0.0101、
  `euclidean_alignment` −0.0124（略降）。
- 按 drift：BN-stats 各级小正向；filter-bank 在 high-drift **−0.024**；EA/RA 各级负向 → **high-drift 受益最小**。
- 这是有价值的 negative/diagnostic 结果，**客观支持**后续学习型适配（online/adapter/prototype），但本阶段不实现/不运行。

**结果在哪看**：`outputs/experiments/alignment_baseline_v1/ALIGNMENT_BASELINE_REPORT.md`（带 headline + 13 节）、
`RUN_STATUS.md`、`cross_session/tables/`（results_alignment_all.csv 等 9 个）、`cross_session/figures/`（6 图）。

**Step 2 铁律**：对齐统计量只能用 train 的数据或 test 的**无标签** X；**绝不用 test session 的 label**。

---

## 4. 明确没做 / 不跑

- ✅ Step 2 adaptation：已完成（结论=无学习对齐不足）。
- ❌ online / test-then-update 未运行、未实现（Step 2 的 negative 结果为其提供依据，但本阶段不做）。
- ❌ 41/10 跨被试预训练、target fine-tuning、LOSO 未运行。
- ❌ CAP-EEGNet full / multi-agent / prototype / memory 未实现（启用 `NotImplementedError`）、未运行。
- ❌ 未改 raw / workspace2 原数据；未覆盖已有 baseline 结果。

---

## 5. 铁律（贯穿全程）

- 路径走 `configs/paths.yaml`，禁硬编码；不写入 raw / workspace2。
- 数据入口 = `eog_ecg_clean` 的 148 个 `status=ok` `.npz`；`derivatives/.mat` 只作标签对照。
- 按被试/session 划分绝不按 trial 泄漏；cross/多源/adaptation 的 val 只能从 train carve，
  **绝不用 test session 的 label**。
- 重任务只走 Slurm（GPU 用 `mi_torch_cu118`），登录节点只做 < ~30s 轻量操作。

> git：之前的工作已提交（HEAD `a239b43` baseline_v1 架构整理）。Step 2 alignment 代码 + 文档在本轮提交。
