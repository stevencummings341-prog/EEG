# BASELINE_PROTOCOL.md

> 基线实验协议说明书（实验规矩，不是创新点）。回答：baseline 怎么跑、哪些已完成、哪些没跑、
> 怎么保证无泄漏、结果放哪里。完整结果分析见 `docs/RESULTS_SUMMARY.md`。

## 1. 当前定位

- 当前 baseline 是 **cross-session DG 的静态 baseline**。
- ✅ 已完成 EEGNet / DeepConvNet / FBCNet 三个模型。
- ✅ 已完成 **within-session 10-fold CV**。
- ✅ 已完成 **single-source cross-session directed pairs**。
- ✅ 已补完 **multi-source `ses-01+02 → ses-03`**。
- ⏳ **Step 2 no-learning adaptation 尚未运行**（见 `docs/ADAPTATION_BASELINE_PLAN.md`）。

## 2. 数据入口

- `processed/eog_ecg_clean`（外部 workspace，路径走 `configs/paths.yaml`）。
- **只使用 `status=ok` sessions**：**148 ok / 5 failed**。
- 5 个 failed：`sub-023/ses-01`、`sub-024/ses-02`、`sub-024/ses-03`、`sub-026/ses-01`、`sub-032/ses-02`。
- 训练**不使用** `derivatives/.mat`（它只作标签对照真值）。
- 每 session 一个 `.npz`：`X[200,58,1000]` µV @ 250Hz，`y[200]∈{0,1}`（1→0 左, 2→1 右）。

## 3. 已完成协议

### A. within-session CV
- 每个 subject/session 内 **Stratified 10-fold**。
- train / val / test **只在同一个 session 内**（val 从 train 折 carve）。
- 用来估计**无跨天漂移的上界**。

### B. single-source cross-session
- **train ses-i, test ses-j**，directed pairs（3 session 全 ok 时 6 个有向对）。
- **val 只从 train session carve**（早停用）。
- **test label 只用于最终 evaluation**。
- 量化跨 session 漂移代价。

### C. multi-source `ses-01+02 → ses-03`（Step 1）
- **train = ses-01 + ses-02**（同被试 trial 合并）。
- **test = ses-03**。
- **只使用三个 session 都 ok 的被试**（47 个；其余跳过并记录原因）。
- **val 只从合并后的 train carve**，绝不含 ses-03。

## 4. 模型

- EEGNet（Lawhern 2018）
- DeepConvNet（Schirrmeister 2017）
- FBCNet（Mane 2021，固定 FIR 滤波器组）
- **CAP-EEGNet / online / prototype / agent 不属于当前 baseline**（future work；CAP-EEGNet v1 在码内但不在当前运行集）。

四个模型共享 `{logits, features, confidence}` forward 契约 + 同一 trainer + 同一指标 → 公平对比。

## 5. 指标

- 分类：`acc`、`bacc`、`f1`、`auc`；校准：`nll`、`brier`、`ece`。
- **每行必须记录 `n_train` / `n_val` / `n_test`**（用于无泄漏核查）。
- 汇总额外给：cross-session drop = within − cross；relative drop = 1 − cross/within；
  multi-source Δ vs best single = multi-source − 最强单源方向。

## 6. 输出路径

| 协议 | 结果目录 | checkpoints | 报告 |
|---|---|---|---|
| within + single-source cross | `outputs/experiments/baseline_v1/provenance/session_model_compare_v1/` | `checkpoints/session_model_compare_v1/{eegnet,deepconvnet,fbcnet}/` | `.../summaries/SESSION_MODEL_COMPARE_REPORT.md` |
| multi-source Step 1 | `outputs/experiments/baseline_v1/provenance/session_multisource_v1/` | `checkpoints/session_multisource_v1/{eegnet,deepconvnet,fbcnet}/` | `.../summaries/MULTISOURCE_STEP1_REPORT.md` |
| 漂移诊断 | `outputs/analysis/session_drift_v1/` | — | `.../SESSION_DRIFT_REPORT.md` |

每个原始 `summaries/` 含 `runs/*.csv`（原始行）、汇总 CSV、图、报告；`splits/` 含 split JSON。

**规范阅读入口（不要再按旧 run_id 理解分类）**：`outputs/experiments/baseline_v1/`

```text
baseline_v1/
├── within_session/                 # within-session CV
├── cross_session/
│   ├── single_source/              # train one source session, test another session
│   └── multi_source/               # train ses-01+ses-02, test ses-03
├── figures/                        # 全部图放一起
└── BASELINE_REPORT.md       # 带图总报告
```

`session_model_compare_v1` 是旧 raw run id，意思是“模型比较运行”，不是一个实验类别；它里面同时包含
within-session 和 single-source cross-session 的原始结果，现归档到
`baseline_v1/provenance/session_model_compare_v1/`。`session_multisource_v1` 是 Step 1 原始运行目录，
现归档到 `baseline_v1/provenance/session_multisource_v1/`。
**文字版整合报告**：`docs/RESULTS_SUMMARY.md`。

## 7. 无泄漏铁律

1. **target / test session 不参与训练、验证、早停、调参**。
2. **test label 只能 final eval**（within 折内 test fold、cross/多源的 test session 的 y）。
3. **split 保存**：within 折索引、cross 有向对、多源每 (subject,seed) 的 train/val 索引都落 JSON。
4. **每行记录 `n_train/n_val/n_test`** 用于检查（within ≈144/36/20；single-source cross 160/40/200；多源 320/80/200）。
5. 同 trainer / 同指标 / 同数据过滤，所有模型一致 → 公平。

## 8. 当前状态

- **static baseline 已完成**（within + single-source cross + multi-source ses-01+02→ses-03）。
- **Step 2 no-learning adaptation 是下一步**（none / session_zscore / Euclidean Alignment /
  Riemannian Alignment / target BN-stats / filter-bank reweighting；见 `docs/ADAPTATION_BASELINE_PLAN.md`）。
- 结果速览见 `docs/RESULTS_SUMMARY.md`；Step 1 深度分析见 `docs/MULTISOURCE_STEP1_REPORT.md`。
