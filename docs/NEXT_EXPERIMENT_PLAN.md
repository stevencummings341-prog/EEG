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

## Step 2 — 下一步：no-learning adaptation baseline（**未运行**）

在 cross-session 协议上做**不更新模型权重**的对齐，量化"零学习成本能挽回多少漂移掉点"。

方法（各跑、互为对照）：
- `none`（对照）
- `session_zscore`
- Euclidean Alignment
- Riemannian Alignment
- target BN statistics adaptation
- filter-bank reweighting

要点：
- models EEGNet/DeepConvNet/FBCNet；seeds 0-4；与 baseline 同配方，单源 cross + 多源都可套用。
- 按 `drift_level`（high/moderate/stable）分组看（Step 1 已发现高漂移被试更需 alignment）。
- **无泄漏铁律**：对齐统计量只能用 train 的数据，或 test 的**无标签** X；**绝不用 test session 的 label**。
- 成功标准：平均 cross Acc 比 `none` ≥ +2pp 且最差方向回升 → 方向对；否则记录"无学习对齐不足"，作为启动 Step 3 依据。
- 代码落点（实现时建）：`src/adaptation/{session_alignment,bn_adaptation}.py`、
  `src/evaluation/session_adaptation_protocols.py`、`scripts/train_session_adaptation.py`、
  `scripts/summarize_adaptation_results.py`、`configs/session_adaptation_compare.yaml`。
  详见 `docs/ADAPTATION_BASELINE_PLAN.md`、`docs/CODE_INTEGRATION_NOTES.md`。

## Step 3 — Future（Step 2 结果出来后才讨论）

online learning · adapter / prototype / memory · CAP-EEGNet full · multi-agent ·
41/10 cross-subject pretraining · target fine-tuning · LOSO。**现在不实现、不跑、不验证。**

---

## 总纪律（每个 Step 都适用）

1. 重任务只走 Slurm，登录节点只做轻量检查；GPU 用 `mi_torch_cu118`，CUDA 不可用 fail-fast。
2. 数据入口 = `eog_ecg_clean` 的 148 个 `status=ok` `.npz`；不写 raw / workspace2。
3. 不按 trial 泄漏；cross/多源/adaptation 的 val 与对齐统计**不得使用 test session 的 label**。
4. 每个新协议先 smoke test 再全量；结果写 `outputs/`，权重写 `checkpoints/`。
5. 文档如实记录"已完成 vs 未运行"，不夸大。
