# ADAPTATION_BASELINE_PLAN.md

> 这是 Step 2 的**计划**。**尚未运行**，绝不能写"已完成"。

## 1. 当前状态

- **Step 2 no-learning adaptation baseline 未运行。**
- 本文档只是计划。
- 上游已完成：static baseline（within + single-source cross + multi-source ses-01+02→ses-03，
  见 `docs/RESULTS_SUMMARY.md` / `docs/MULTISOURCE_STEP1_REPORT.md`）。

## 2. 目标问题

- static baseline 已证明 **cross-session drop**（约 9–13%）。
- multi-source 已证明**多源训练能缓解一部分**（三模型均优于最强单源，均值 +0.0281）。
- Step 2 要验证：**不用 target label，只用无标签统计对齐，能不能进一步缓解 drift。**

## 3. 方法列表

- `none`（对照）
- `session_zscore`
- Euclidean Alignment (EA)
- Riemannian Alignment
- BN statistics adaptation
- filter-bank reweighting

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

## 6. 方法细节

**A. z-score** — session / channel-wise mean、std 归一化。
**B. EA（Euclidean Alignment）** — covariance mean + inverse sqrt（白化到单位协方差）。
**C. Riemannian** — SPD mean（黎曼均值）做对齐，**优先 numpy/scipy 自实现，不安装 pyriemann**。
**D. BN stats** — target X forward 更新 BN running mean/var，**不反向传播**。
**E. filter-bank** — μ/β band power 统计或频带权重重加权。

## 7. 输出路径建议

- `outputs/experiments/session_adaptation_v1/`（runs / splits / summaries / figures）
- `checkpoints/session_adaptation_v1/`

## 8. 验收标准

- **smoke test 先过**（少被试 / 少 epoch / GPU srun）。
- **full run 后和 `none` baseline 对比**。
- 报告包含：**方向表**（6 个有向 + 多源）、**模型表**（mean±std）、**per-subject 表**（含 drift_level 分层）。
- 成功 = 平均 cross Acc 比 `none` ≥ +2pp 且最差方向回升；否则记录"无学习对齐不足"，作为启动 Step 3 依据。

## 9. 代码落点（实现时建，本计划不创建）

见 `docs/CODE_INTEGRATION_NOTES.md` §4。核心：`src/adaptation/{session_alignment,bn_adaptation}.py`、
`src/evaluation/session_adaptation_protocols.py`、`scripts/train_session_adaptation.py`、
`scripts/summarize_adaptation_results.py`、`configs/session_adaptation_compare.yaml`。
