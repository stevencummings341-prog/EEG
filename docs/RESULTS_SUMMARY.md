# 结果总结报告 (Results Summary) — cross-session DG 主线【整合版】

> **这是 cross-session DG 静态 baseline 的单一整合报告**：session 漂移诊断 + 三个 baseline
> （EEGNet / DeepConvNet / FBCNet）在 **within-session CV + single-source cross-session +
> Step 1 multi-source `ses-01+02 -> ses-03`** 下的 5-seed 评估，并**重点分析"为什么比论文低"**。
> 数据入口：`eog_ecg_clean` 的 `status=ok` session（148 ok / 5 failed）。
>
> **图集（全部 18 张放一起）+ 静态 baseline 总报告**：`outputs/experiments/static_baseline_v1/`
> （`figures/` + `STATIC_BASELINE_REPORT.md`）。
> 原始详表：`outputs/experiments/session_model_compare_v1/summaries/`（within+cross）、
> `outputs/experiments/session_multisource_v1/summaries/`（Step 1）、
> `outputs/analysis/session_drift_v1/`（drift）。
> Step 1 深度分析（per-subject / drift-level）见 `docs/MULTISOURCE_STEP1_REPORT.md`。
> **Step 2 no-learning adaptation 未运行**（`docs/ADAPTATION_BASELINE_PLAN.md`）。

---

## 1. 做了什么（一句话）

1. **Session 漂移诊断**（数据层面，非训练）：144 个被试内 session pair / 50 被试，量化跨 session 漂移。
2. **三个 baseline 统一比较**：EEGNet / DeepConvNet / FBCNet，同一协议/数据过滤/指标，5 个 seed（0–4），
   两个协议 **Within-session 10-fold CV** 与 **Cross-session**（同被试有向 session 对）。
3. 共 **26 520** 次训练（within 22 200 + single-source cross 4320），30/30 (model×protocol×seed) cells 全完成，无泄漏、无 NaN。
4. **Step 1 multi-source baseline 已补齐**：`ses-01+02 -> ses-03`，47 个三 session 全 ok 被试，
   3 models × 5 seeds × 47 subjects = **705** rows，0 failed，0 NaN。

CAP-EEGNet 与 agent/toolkit/prototype/confidence/online/fine-tuning、LOSO、41/10 均为 **future work，未跑**。

---

## 2. Session 漂移诊断结果（direction A）

| 指标 | 均值 | 解读 |
|---|---|---|
| MMD | 0.238 | 整体存在**中等**分布漂移 |
| CSP similarity | 0.420 | 空间判别模式只**中等稳定** |
| ERD/ERS μ / β corr | 0.419 / 0.482 | 感觉运动节律空间模式**有明显漂移** |
| μ-KS | 0.246 | μ 频段功率分布发生可观变化 |
| RMS ratio (median) | 0.992 | 整体幅值**接近稳定**（不是主因）|
| Fisher shift | ≈ 0 | 左右手可分性无系统增强/减弱 |

- 按 session pair：**1-3 整体 MMD 最大**（相隔最远），**2-3 在 μ-KS/CSP/ERD 上最稳定**（部分支持学习效应）。
- 结论：跨 session 漂移**主要是空间模式 + μ/β 频谱分布的变化**，不是简单幅值缩放。
- 详见 `outputs/analysis/session_drift_v1/SESSION_DRIFT_REPORT.md`（含按 pair、按被试分析与图）。

---

## 3. Baseline 结果（5 seed，mean ± std）

### Within-session 10-fold CV（上界，无漂移）

| 模型 | Acc | BalAcc | MacroF1 | AUC | ECE |
|---|---:|---:|---:|---:|---:|
| **EEGNet** | **0.807 ± 0.002** | 0.807 | 0.803 | 0.859 | 0.142 |
| DeepConvNet | 0.766 ± 0.002 | 0.766 | 0.759 | 0.825 | 0.171 |
| FBCNet | 0.720 ± 0.003 | 0.720 | 0.714 | 0.772 | 0.190 |

### Cross-session（同被试 train ses-i → test ses-j）

| 模型 | Acc | BalAcc | MacroF1 | AUC |
|---|---:|---:|---:|---:|
| **EEGNet** | **0.711 ± 0.008** | 0.711 | 0.705 | 0.761 |
| DeepConvNet | 0.681 ± 0.002 | 0.681 | 0.666 | 0.741 |
| FBCNet | 0.628 ± 0.003 | 0.628 | 0.594 | 0.690 |

### Within vs Cross（漂移代价）

| 模型 | within | cross | drop | relative drop |
|---|---:|---:|---:|---:|
| EEGNet | 0.807 | 0.711 | 0.096 | **11.9%** |
| DeepConvNet | 0.766 | 0.681 | 0.085 | 11.1% |
| FBCNet | 0.720 | 0.628 | 0.092 | 12.8% |

→ 三个 baseline 跨 session 一致掉 **9–13%**，量化了 session 漂移的代价；排序 EEGNet > DeepConvNet > FBCNet。

### Step 1 Multi-source: `ses-01+02 -> ses-03`（已补齐）

| 模型 | ses-01→03 | ses-02→03 | **ses-01+02→03** | Δ vs best single |
|---|---:|---:|---:|---:|
| EEGNet | 0.6991±0.009 | 0.7492±0.008 | **0.7717±0.003** | +0.0224 |
| DeepConvNet | 0.6757±0.004 | 0.7211±0.009 | **0.7564±0.007** | +0.0353 |
| FBCNet | 0.6142±0.006 | 0.6484±0.005 | **0.6750±0.002** | +0.0267 |

Step 1 结论：把 `ses-01` 与 `ses-02` 合并训练后，三个模型都超过最强单源
`ses-02→03`；平均 Δ vs best single = **+0.0281**。说明在静态 baseline 内，多一天 source
session 能稳定提升 `ses-03` 泛化。无泄漏检查：`n_train=320`、`n_val=80`、`n_test=200`，
val 只从 `ses-01+02` carve，`ses-03` label 仅用于 final evaluation。

---

## 4. 为什么比论文低？（核心问题）

论文 within-session 10-fold CV：EEGNet 85.32% / DeepConvNet 84.47% / FBCNet 78.40%。
我们：80.67% / 76.63% / 72.03%，分别低 **4.65 / 7.84 / 6.37 pp**。

### 4.1 先给结论：**不是模型架构写错了**

- EEGNet 严格按 Lawhern et al. 2018 复刻（temporal conv → depthwise spatial → separable，
  F1=8/D=2/F2=16/K=64），forward 形状、特征维度都验证过。
- **三个模型的 gap 方向一致、排序与论文一致（EEGNet>DeepConvNet>FBCNet）、session 学习趋势也复现
  （S1<S2<S3：EEGNet 77.9→81.7→82.3%）**。如果是某个架构写错，应表现为该模型异常崩塌或排序错乱，
  而不是三者**一致地**低 ~5–8pp。这强烈指向**系统性的训练配方/数据预算差异**，而非单模型 bug。

### 4.2 主要原因（按影响从大到小）

1. **within 每折又切了 20% 验证集做早停 → 训练样本变少。**
   每 session 200 trial，10-fold → 测试 20、训练 180；我们再从 180 里切 `val_fraction=0.2`=36 做早停，
   **实际每折只用 144 个 trial 训练**（论文 10-fold 用满 180）。少 ~20% 训练数据，对这种每 session 仅
   200 trial 的小数据集，深度网络精度对训练量很敏感 —— 这是**最大的单一原因**。

2. **早停信号噪声大。** 验证集只有 ~36 个样本，`val_loss` 在如此小的集合上波动大，
   `patience=20` 仍可能**过早停止**或选到次优 epoch。

3. **正则化偏弱 / 与论文不同。** EEGNet `dropout=0.25`（within-subject 常用 0.5）、`weight_decay=0`，
   且**三个模型都没有实现原论文的 max-norm 权重约束**。在 144 样本上更易过拟合或欠收敛。

4. **统一配方、未按模型调参（为公平比较）。** 所有模型共用 `bs=16, lr=1e-3, Adam, max_epochs=100`，
   没有按模型单独调 epoch/学习率/增强。FBCNet 还做了简化实现（固定 FIR 滤波器组、省略 max-norm 与部分细节），
   DeepConvNet 也没有逐模型调优 —— 这解释了为何 DeepConvNet/FBCNet 的 gap 比 EEGNet 略大。

5. **预处理差异（次要）。** 我们的 `eog_ecg_clean` 比论文多做了一步 EOG/ECG ICA 去伪迹，QC 显示 μ 带功率
   中位数约为官方的 0.898（判别信息基本保留，见 `QC_SUMMARY_CN.md`），可能让可分性略降一点点。

6. **论文口径可能不同（无法完全对齐）。** epoch 预算、数据增强、是否报告 best-epoch、随机划分细节等
   论文未完全公开，难 1:1 复现。

### 4.3 关键：这**不影响本研究的结论**

我们要回答的是**“within vs cross 的 accuracy drop”**，而上述配方对 within 和 cross 是**完全一致施加**的
（同 trainer、同 val 切分、同早停、同超参）。因此**跨 session drop（9–13%）这个核心发现是公平、可靠的**，
绝对精度比论文低不改变“跨 session 显著掉点”的结论。

### 4.4 如果想把绝对精度贴近论文（可选，非必需）

- within 协议**取消额外的 val 切分**（用固定 epoch 训练，或在折内用全部 180 trial），是预计提升最大的一步；
- `dropout 0.25 → 0.5`、加 **max-norm** 约束、适当 `weight_decay`；
- 按模型单独调 epoch / 增强（time shift、噪声、mixup 等）；
- 注意：若为了对齐论文而改配方，应**within 与 cross 同步改**，以保持比较公平。

---

## 5. 可靠性与无泄漏（已核验）

- Completed cells：**30/30**（3 模型 × 2 协议 × 5 seed）。
- within：用满 **148/148** ok session；cross：**288/288** 有向对；same-session 非法对 = 0。
- Step 1 multi-source：47 个三 session 全 ok 被试；跳过 4 个不完整被试
  (`sub-023/sub-024/sub-026/sub-032`)；705/705 rows ok。
- NaN 指标 = 0。
- 无泄漏（代码层面保证）：within 折内 train/test 不相交、验证集只从 train 切；single-source cross
  train/test 是不同 session；multi-source Step 1 的 val 只从 `ses-01+02` carve，`ses-03`
  从不参与训练/验证/早停。

---

## 6. 结论

1. **EEGNet 是最强 baseline**（within 80.7% / cross 71.1%），DeepConvNet 次之，FBCNet 最低 —— 与论文排序一致。
2. **跨 session 一致掉 9–13%**，且漂移诊断显示主因是**空间模式 + μ/β 频谱漂移**（幅值稳定、可分性无系统变化）。
3. **Step 1 multi-source (`ses-01+02 -> ses-03`) 已补齐且有效**：三个模型均优于最强单源 `ses-02→03`。
4. 与论文的 ~5–8pp 绝对差距来自**训练配方/数据预算**（尤其 within 折内再切验证集），**不是架构 bug**；
   且不影响 within-vs-cross 的核心结论。

---

## 7. 下一步

- （可选）如要对齐论文绝对精度：按 §4.4 调整 within 配方并 within/cross 同步重跑。
- **Step 2（下一步，未运行）**：no-learning adaptation baseline。依据漂移诊断，优先 **spatial alignment（Euclidean Alignment / CORAL）**、
  **frequency / filter-bank adaptation**、**BN/adapter**，而不是只做全局幅值归一化。
- CAP-EEGNet 及其复杂模块（prototype / 多源 confidence / online / fine-tuning）仍为 future work。

---

## 8. 文件索引

| 内容 | 路径 |
|---|---|
| **整合图集（全部 18 张）** | `outputs/experiments/static_baseline_v1/figures/`（`drift_*` / `within_*` / `cross_session_*`） |
| **静态 baseline 总报告** | `outputs/experiments/static_baseline_v1/STATIC_BASELINE_REPORT.md` |
| Baseline 主报告 | `outputs/experiments/session_model_compare_v1/summaries/SESSION_MODEL_COMPARE_REPORT.md` |
| Baseline 汇总表 | `.../summaries/summary_by_model_protocol.csv`、`within_by_seed.csv`、`cross_by_seed.csv`、`within_session_wise.csv`、`cross_by_direction.csv` |
| Baseline 原始行 | `.../summaries/results_within_session.csv`、`results_cross_session.csv` |
| Baseline 图 | `.../summaries/{within_session_accuracy_boxplot,cross_session_accuracy_matrix_by_model,protocol_comparison}.png` |
| Step 1 multi-source 报告 | `docs/MULTISOURCE_STEP1_REPORT.md`、`outputs/experiments/session_multisource_v1/summaries/MULTISOURCE_STEP1_REPORT.md` |
| Step 1 multi-source 表/图 | `outputs/experiments/session_multisource_v1/summaries/{results_multisource_0102_to_03,multisource_by_seed,multisource_by_model,summary_by_model_protocol}.csv`、`outputs/experiments/session_multisource_v1/figures/multisource_vs_singlesource_acc.png` |
| 运行状态 | `.../summaries/RUN_STATUS.md` |
| 漂移报告 | `outputs/analysis/session_drift_v1/SESSION_DRIFT_REPORT.md`（+ `SESSION_DRIFT_SUMMARY_CN.md`） |
| 漂移表/图 | `.../session_drift_v1/{session_pair_summary,per_subject_drift_summary}.csv`、`figures/` |
| 协议定义 | `docs/BASELINE_PROTOCOL.md`、`docs/SESSION_DRIFT_ANALYSIS.md` |
