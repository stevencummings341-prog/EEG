# ADAPTATION_BASELINE_PLAN.md

> Step 2 计划 + 当前进度。**代码实现 + smoke 通过 + full run 已提交，结果 PENDING**。
> summarizer 写出 `results_alignment_all.csv` 之前，**绝不能写"已完成"**。

## 1. 当前状态（2026-06-09）

- **Step 2 no-learning adaptation baseline：代码已实现、smoke 已通过、full run 已提交、summarizer
  依赖已提交，结果未跑完（PENDING）。** 不是已完成。
- Slurm：75 GPU train jobs `21261-21335` + CPU summarizer `21336`（`afterany`）。
  job ids → `outputs/experiments/alignment_baseline_v1/full_job_ids.txt`。
- 输出目录：`outputs/experiments/alignment_baseline_v1/`（结果 + 报告待 summarizer 跑完）。
- checkpoints：`checkpoints/alignment_baseline_v1/`。
- 实现细节：见 §6 与 `docs/PROGRESS.md`(2026-06-09 Step 2 条目)；none_reference 取自 baseline_v1，不重跑。
- 上游已完成：static baseline（within + single-source cross + multi-source ses-01+02→ses-03，
  见 `docs/RESULTS_SUMMARY.md` / `docs/MULTISOURCE_STEP1_REPORT.md`）。

## 2. 目标问题

- static baseline 已证明 **cross-session drop**（约 9–13%）。
- multi-source 已证明**多源训练能缓解一部分**（三模型均优于最强单源，均值 +0.0281）。
- Step 2 要验证：**不用 target label，只用无标签统计对齐，能不能进一步缓解 drift。**

## 3. 方法列表（实现名）

- `none_reference`（对照，取自 baseline_v1，不重跑）
- `session_zscore`
- `euclidean_alignment` (EA)
- `riemannian_alignment`（log-Euclidean SPD mean）
- `bn_statistics_adaptation`
- `filterbank_reweighting`

## 4. 统一协议

- **source session 训练模型**。
- **target / test session 只允许使用 X 计算无标签统计量**。
- **`y_test` 只能 final evaluation**。
- **不允许用 target label 训练、早停、调参**。
- **除 BN running stats 外，不允许在 target 上 `optimizer.step`**（不更新可训练权重）。

> 这是 Step 2 的公平性边界：**允许用 test 的 X，不允许用 test 的 y。**

## 5. 适用场景

- **single-source**：`ses-01→02`、`ses-01→03`、`ses-02→03` 等有向对。
- **multi-source**：`ses-01+02→ses-03`。
- 优先和**已有 baseline 对齐**（同 split / 同 seed / 同 trainer / 同指标），直接对比 `none`。

## 6. 方法细节（实现）

**A. session_zscore** — channel-wise mean/std 归一化；source 统计量只来自 source train，应用到
source train+val；target 用 target X 无标签统计量。
**B. euclidean_alignment** — 每 trial 协方差的算术均值 R，`R^{-1/2}` 经对称 eigh 白化到单位协方差；
R 先加 `eps` ridge + 对角 shrinkage（向 tr(R)/C·I），特征值 clip>0，避免奇异。source R 来自 source
train，target R 来自 target X。矩阵 58×58。
**C. riemannian_alignment** — `G = expm(mean_i logm(C_i))` 的 **log-Euclidean SPD mean**，
`G^{-1/2}` 白化；logm/expm 均经对称 eigh + 特征值 clip 实现，**numpy/scipy 自实现，不安装 pyriemann**
（若环境已有则仅记录 `pyriemann_available`，不调用）。矩阵 58×58。
**D. bn_statistics_adaptation** — source 训练后复制 best model，对 target X forward 1 遍只刷新 BN
running mean/var（`reset=True, momentum=None` 累积均值），**不算 loss、不 backward、无 optimizer**。
**E. filterbank_reweighting** — θ(4-8)/μ(8-13)/β(13-30)/low-γ(30-40) 的固定 FIR 子带，按 source 频带
功率 profile 把 target 各带功率重加权（单标量 gain，clip 到 [0.5,2.0]）；保守版，仅用 target X 估功率。

每行记录 `source_alignment_stats` / `target_alignment_stats`（含 eps/shrinkage/矩阵形状/频带功率/权重等）、
`used_target_x_for_stats`（trained 方法 True；none_reference False）、`used_target_y_for_training`（恒 False）。

## 7. 输出路径（实际）

- `outputs/experiments/alignment_baseline_v1/`（README / ALIGNMENT_BASELINE_REPORT.md / RUN_STATUS.md /
  manifest_sources.json / resolved_config_summary.yaml / RUN_PLAN.md / full_job_ids.txt /
  `cross_session/{runs,splits,tables,figures}`）
- `checkpoints/alignment_baseline_v1/{method}/{model}/`

## 8. 验收标准

- **smoke test 先过**（少被试 / 少 epoch / GPU srun）。
- **full run 后和 `none` baseline 对比**。
- 报告包含：**方向表**（6 个有向 + 多源）、**模型表**（mean±std）、**per-subject 表**（含 drift_level 分层）。
- 成功 = 平均 cross Acc 比 `none` ≥ +2pp 且最差方向回升；否则记录"无学习对齐不足"，作为启动 Step 3 依据。

## 9. 代码落点（已实现）

`src/adaptation/{__init__,session_alignment,bn_adaptation}.py`、
`src/evaluation/session_alignment_protocols.py`、
`scripts/{train_session_alignment,summarize_alignment_results,build_alignment_baseline_outputs}.py`、
`configs/session_alignment_compare.yaml`、
`scripts/slurm/{train_session_alignment_gpu,summarize_alignment_results_cpu}.sbatch`。
复用：`src/training/trainer.py`、`src/models/registry.py`、`src/data/session_splits.py`、
`src/evaluation/metrics.py`、baseline 的 stratified-val 切分逻辑。
