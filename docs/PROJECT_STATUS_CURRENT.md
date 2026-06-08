# PROJECT STATUS — CURRENT（权威状态页 / single source of truth）

> 读这一份就能对齐"现在做到哪、下一步做什么"。更细：逐日 `docs/PROGRESS.md`；
> 全量总览 `docs/PROJECT_OVERVIEW.md`；结果 `docs/RESULTS_SUMMARY.md`；
> Step 1 深度报告 `docs/MULTISOURCE_STEP1_REPORT.md`；下一步计划 `docs/NEXT_EXPERIMENT_PLAN.md` +
> `docs/ADAPTATION_BASELINE_PLAN.md`；P10 整合 `docs/P10_INTEGRATION_SUMMARY.md`。
>
> 最后更新：2026-06-08（Step 1 完成）；2026-06-09 文档系统性修复。

---

## 0. 一句话现状

项目主线 = **EEG MI 跨 session 域泛化（cross-session domain generalization）**。
已完成：漂移诊断 → 静态 baseline（within / 单源 cross / **多源 cross**）。
**下一步 = Step 2 no-learning adaptation baseline（未运行）**。online / 41-10 / fine-tuning /
CAP-EEGNet full / multi-agent / prototype / memory 均为 future，未运行、未验证。

---

## 1. 主线进度

| Step | 内容 | 状态 |
|---|---|---|
| A | Session drift 诊断（144 pairs / 50 subjects，9 指标） | ✅ 已完成 |
| B | 静态 baseline：within-session 10-fold CV + 单源有向 cross-session（EEGNet/DeepConvNet/FBCNet，5 seeds，26 520 trainings，30/30 cells，无泄漏/NaN） | ✅ 已完成 |
| Step 1 / C | 静态 baseline 补齐：**multi-source `ses-01+ses-02 → ses-03`**（47 被试，4 跳过，705 rows） | ✅ 已完成 |
| Step 2 / D | **no-learning adaptation baseline**（none / session_zscore / Euclidean Alignment / Riemannian Alignment / target BN-stats / filter-bank reweighting） | 🔜 下一步，**未运行** |
| Step 3+ | online / adapter / prototype / memory / CAP-EEGNet full / 41-10 / fine-tuning / LOSO | 🚧 future，未运行/未验证 |

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
  `outputs/experiments/cross_session_baseline_v1/`（`figures/` 全部 18 张图集 + `INTEGRATED_REPORT.md` 带图报告）
- `outputs/experiments/session_model_compare_v1/summaries/`（within + single-source cross 原始）
- `outputs/experiments/session_multisource_v1/summaries/`（Step 1 原始 + `multisource_by_subject.csv`）
- `outputs/analysis/session_drift_v1/`（drift 原始）

---

## 3. 下一步（Step 2，未运行）

no-learning adaptation baseline（不更新模型权重）：`none` / `session_zscore` / Euclidean Alignment /
Riemannian Alignment / target BN statistics adaptation / filter-bank reweighting。
细节与无泄漏规则见 `docs/ADAPTATION_BASELINE_PLAN.md`，代码落点见 `docs/CODE_INTEGRATION_NOTES.md`。

**Step 2 铁律**：对齐统计量只能用 train 的数据或 test 的**无标签** X；**绝不用 test session 的 label**。

---

## 4. 明确没做 / 不跑

- ❌ Step 2 adaptation 未运行。
- ❌ online / test-then-update 未运行、未实现。
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

> ⚠️ git：HEAD 仍是 2026-06-04 scaffold，之后工作均未提交。建议尽快 commit（见 `docs/PROGRESS.md` 2026-06-09）。
