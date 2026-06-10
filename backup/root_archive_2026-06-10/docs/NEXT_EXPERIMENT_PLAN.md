# NEXT EXPERIMENT PLAN — Step 0/1/2/3

> 当前权威状态见 `docs/PROJECT_STATUS_CURRENT.md`；Step 1 报告 `docs/MULTISOURCE_STEP1_REPORT.md`；
> Step 2 细节 `docs/ADAPTATION_BASELINE_PLAN.md`；代码落点 `docs/CODE_INTEGRATION_NOTES.md`。

---

## Step 0 — Docs / P10 integration ✅ 完成

统一方向口径，把学长 P10 材料整合进主项目认知；建立状态页/计划/参考索引。无实验。

## Step 1 — Static baseline 补齐：multi-source `ses-01+02 → ses-03` ✅ 完成（2026-06-08）

- protocol：同一被试 train=ses-01+ses-02、test=ses-03，val 只从合并 train carve，不用 ses-03 label。
- models EEGNet/DeepConvNet/FBCNet；seeds 0,1,2,3,4。
- used subjects = 47（三 session 全 ok）；skipped = 4（`sub-023`/`sub-024`/`sub-026`/`sub-032`）。
- Slurm：train `21240-21244` + summarize `21245`，全部 COMPLETED；705 rows ok / 0 failed / 0 NaN。
- 结果：

  | model | ses-01+02→ses-03 | Δ vs best 单源 |
  |---|---:|---:|
  | EEGNet | 0.7717±0.003 | +0.0224 |
  | DeepConvNet | 0.7564±0.007 | +0.0353 |
  | FBCNet | 0.6750±0.002 | +0.0267 |

- 核心发现：multi-source 对三模型都优于最强单源 `ses-02→03`；平均 +0.0281；
  失败案例集中在两 source 质量差异大的被试 → 动机 Step 2。完整分析 `docs/MULTISOURCE_STEP1_REPORT.md`。

## Step 2 — no-learning adaptation baseline ✅ 完成（2026-06-09，结论=无学习对齐不足）

在 cross-session 协议上做**不更新模型权重**的对齐，量化"零学习成本能挽回多少漂移掉点"。

**结果（诚实）**：75 GPU jobs `21261-21335` + summarizer `21336` 全部 COMPLETED；
`results_alignment_all.csv` 30,150 rows，0 failed / 0 NaN。mean Δacc vs none_reference：

| method | Δacc | |
|---|---:|---|
| `bn_statistics_adaptation` | **+0.0071** | 唯一正向，但 < +2pp |
| `filterbank_reweighting` | −0.0030 | 近中性 |
| `session_zscore` | −0.0038 | 近中性 |
| `riemannian_alignment` | −0.0101 | 略降 |
| `euclidean_alignment` | −0.0124 | 降最多 |

绝对 acc：none_reference 0.6818、bn_statistics 0.6889。按 drift：BN-stats 各级小正向；filter-bank
在 high-drift −0.024；EA/RA 各级负向（high-drift 受益最小）。**没有任何方法达到 +2pp 成功线
→ 记录为"无学习统计对齐不足"**，构成 Step 3 学习型适配的客观依据（本阶段不实现/不运行）。
报告：`outputs/experiments/alignment_baseline_v1/ALIGNMENT_BASELINE_REPORT.md`。

### 方法/协议（已实现，备查）

方法（各跑、互为对照）：
- `none_reference`（对照，取自 baseline_v1，不重跑）
- `session_zscore`
- Euclidean Alignment
- Riemannian Alignment（**log-Euclidean SPD mean**，numpy/scipy 自实现，不依赖 pyriemann）
- target BN statistics adaptation（仅刷新 BN running stats，无 backward/optimizer）
- filter-bank reweighting（θ/μ/β/low-γ FIR 频带功率重加权，保守标量版）

进度：
- models EEGNet/DeepConvNet/FBCNet；seeds 0-4；与 baseline 同配方。协议 = 单源 6 有向对（288）+
  多源 `ses-01+02→ses-03`（47）。Est. 25,125 trainings。
- Slurm：75 GPU train jobs `21261-21335`（method × model × seed）+ summarizer `21336`
  （`afterany`）。job ids 见 `outputs/experiments/alignment_baseline_v1/full_job_ids.txt`。
- **无泄漏铁律**：对齐统计量只能用 train 数据或 test 的**无标签** X；**绝不用 test session 的 label**；
  每行记录 `used_target_x_for_stats`（trained 方法 True）+ `used_target_y_for_training`（恒 False）。
- 成功标准：平均 cross Acc 比 `none_reference` ≥ +2pp 且最差方向回升 → 方向对；否则记录"无学习对齐不足"，
  作为启动 Step 3 依据。按 `drift_level`(high/moderate/stable) 分层分析（summarizer 已实现）。
- 代码落点（已建）：`src/adaptation/{session_alignment,bn_adaptation}.py`、
  `src/evaluation/session_alignment_protocols.py`、`scripts/{train_session_alignment,
  summarize_alignment_results,build_alignment_baseline_outputs}.py`、
  `configs/session_alignment_compare.yaml`、`scripts/slurm/*alignment*`。
- **结果待跑完**：回来看 `sacct -j 21261-21336`、`.../alignment_baseline_v1/RUN_STATUS.md`、
  `.../cross_session/tables/results_alignment_all.csv`、`.../ALIGNMENT_BASELINE_REPORT.md`。
  summarizer 写出 `results_alignment_all.csv` 后才算 Step 2 完成。

## Step 3 — Future（Step 2 已给出客观依据；现仍不实现/不跑）

Step 2 已证明无学习统计对齐不足、high-drift 被试受益最小 → **学习型 target 适配**是合理下一步：
online test-then-update · adapter / prototype / memory · CAP-EEGNet full · multi-agent ·
41/10 cross-subject pretraining · target fine-tuning · LOSO。**现在不实现、不跑、不验证。**
若做，建议从最轻量的开始（如把 BN-stats 作默认前端 + 轻量 adapter），并优先针对 high-drift 被试。

---

## 总纪律（每个 Step 都适用）

1. 重任务只走 Slurm，登录节点只做轻量检查；GPU 用 `mi_torch_cu118`，CUDA 不可用 fail-fast。
2. 数据入口 = `eog_ecg_clean` 的 148 个 `status=ok` `.npz`；不写 raw / workspace2。
3. 不按 trial 泄漏；cross/多源/adaptation 的 val 与对齐统计**不得使用 test session 的 label**。
4. 每个新协议先 smoke test 再全量；结果写 `outputs/`，权重写 `checkpoints/`。
5. 文档如实记录"已完成 vs 未运行"，不夸大。
