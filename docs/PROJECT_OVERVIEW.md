# 项目总览与运行指南 (Project Overview & Run Guide)

> 本文件是给你（和未来的 AI 助手）看的"一份读懂全部"的总文档：项目架构细节、
> 每个部分做了什么、目前做到哪、以及在当前架构下**如何一步步跑完整个实验流程**。
> 更短的固定约束在 `.cursor/rules/`；逐日进度在 `docs/PROGRESS.md`；
> 项目宗旨来源在 `docs/references/ChatGPT-EEG-MI-pretraining.md`（学长聊天记录）。
>
> **当前主线（2026-06-06 起，按学长 P10 方案调整）= 预处理后数据的「跨 session 域泛化」研究。**
> 不再继续 41/10 跨被试预训练主线，近期**不跑** 41/10 / LOSO / 微调 / online（全部转为 future work）。
> 做四件事：**A** session 漂移诊断（`docs/SESSION_DRIFT_ANALYSIS.md`）、**B** 统一比较框架
> （EEGNet/DeepConvNet/FBCNet baseline + 我们的 CAP-EEGNet v1）、**C** 在 **Within-session CV** 与
> **Cross-session** 两个层面对比、**D** 统一结果表/图/报告（`docs/BASELINE_PROTOCOL.md`、
> `scripts/summarize_session_results.py`）。数据入口只用 `eog_ecg_clean` 的 `status=ok` 148 个 session。
>
> **进度更新（2026-06-08）：** A 漂移诊断 ✅；B 静态 baseline（within CV + **单源**有向 cross）✅；
> **Step 1 = multi-source `ses-01+ses-02 → ses-03` ✅ 已完成**（47 被试，4 跳过，705 rows；multi-source
> 优于最强单源，EEGNet 0.7717 vs 0.7492，详见 `docs/MULTISOURCE_STEP1_REPORT.md`）。
> **下一步 = Step 2 no-learning adaptation baseline**（none / session_zscore / Euclidean / Riemannian /
> target BN-stats / filter-bank reweighting），**尚未运行**（计划见 `docs/ADAPTATION_BASELINE_PLAN.md`）。
> CAP-EEGNet 当前是 **v1**（encoder + 分类头 + 学习型置信度头），原型/多源置信度/adapter/域对齐/在线均为
> future（见 `docs/ROADMAP.md`）。**权威状态页：`docs/PROJECT_STATUS_CURRENT.md`。** P10 整合见
> `docs/P10_INTEGRATION_SUMMARY.md`。GPU 环境：`mi_torch_cu118`（torch 2.7.1+cu118）。
> 上游不变：全量 51×3 `eog_ecg_clean` 预处理完成（148 ok / 5 failed，QC PASS）。逐日见 `docs/PROGRESS.md`。

---

## 0. 一句话项目定义

基于上海大学 **WBCIC-SHU 2C** 运动想象 EEG 数据集，用 **纯 Python/PyTorch** 从 raw BDF
开始预处理，研究 **跨 session 域泛化（cross-session domain generalization）**：先诊断漂移，
再建静态 baseline（within / 单源 cross / 多源 cross），再做 no-learning adaptation，最后才考虑
在线持续学习。带显式的**样本置信度**是长期目标的一部分。

> 长期愿景（按学长聊天记录，见 `docs/ROADMAP.md`）：**Confidence-aware Online Adaptive
> Multi-Subagent Pretraining Framework for Cross-subject MI EEG Decoding**（置信度感知 +
> 在线自适应 + 多神经子模块 的跨被试 MI 框架）。当前跑通的 EEGNet/DeepConvNet/FBCNet 与
> minimal/v1 CAP-EEGNet 是 baseline，full CAP-EEGNet（多神经子模块、多源置信度、多级原型、
> adapter、域对齐、在线更新）仍是 **future**。

- MATLAB(`preprocessed.m`)/Neuracle 工具箱**只作配方参考，不是运行依赖**。
- `derivatives/` 里论文已处理的 `.mat` **只作标签对照真值，不作训练数据入口**。
- 训练数据入口 = 我们自己 Python 预处理出的 **每 session 一个 `.npz`**（正式 eog_ecg_clean）。

---

## 1. 当前完成度一览

| 模块 | 状态 | 说明 |
|---|---|---|
| 目录/规则/配置/文档 框架 | ✅ 完成 | 路径管理、manifest、规则、文档齐备 |
| 路径管理（外置 + 校验） | ✅ 完成 | `configs/paths.yaml` + `src/utils/paths.py` |
| raw 数据 manifest | ✅ 完成 | 51 被试 / 153 session / 0 缺失 |
| evt.bdf 事件解析（TAL） | ✅ 完成 | `src/preprocessing/neuracle_events.py` |
| 正式预处理 eog_ecg_clean → `.npz` | ✅ 完成 | `eog_ecg_clean.py` + `pipeline.py` |
| 全量 51×3 预处理 | ✅ 完成 | 153 session → **148 ok / 5 failed**（6.15 GiB） |
| 全量质检 + 对比官方 derivatives | ✅ 完成并通过 | `qc_vs_derivatives/`（QC PASS） |
| Dataset / DataLoader | ✅ 完成 | `src/data/shu_dataset.py`（`SHUTrialDataset`） |
| **A. Session 漂移诊断** | ✅ 完成 | 144 pairs / 50 subjects；`src/analysis/session_drift.py` |
| **B. 静态 baseline（within CV + 单源 cross）** | ✅ 完成 | EEGNet/DeepConvNet/FBCNet，5 seeds，26 520 trainings |
| **Step 1 / C. 多源 cross（ses-01+02→ses-03）** | ✅ 完成 | 47 被试，705 rows；`docs/MULTISOURCE_STEP1_REPORT.md` |
| **Step 2. no-learning adaptation baseline** | 🔜 下一步，未跑 | EA/Riemannian/BN-stats/filter-bank/zscore；`docs/ADAPTATION_BASELINE_PLAN.md` |
| CAP-EEGNet（v1：encoder+分类头+学习型 confidence） | ✅ 完成 | `src/models/cap_eegnet.py`（baseline，未列入当前主线运行） |
| CAP-EEGNet（full：subagents/prototype/adapter/domain/online） | 🚧 骨架 | 启用会 `NotImplementedError`；future |
| 在线 / 41-10 / 微调 / LOSO / multi-agent / memory | 🚧 future | 设计/草稿，未运行、未验证 |
| 评估指标 | ✅ 完成 | `src/evaluation/metrics.py`（acc/bacc/f1 + auc/ece/nll/brier） |

图例：✅ 已完成可用 / 🔜 下一步（计划，未跑）/ 🚧 骨架或 future（未运行/未验证）。

---

## 1.5 当前主线文件（cross-session DG）

| 类别 | 文件 | 作用 |
|---|---|---|
| 配置 | `configs/session_drift.yaml` | 漂移诊断参数 + 输出位置 + smoke 子集 |
| 配置 | `configs/session_model_compare.yaml` | within/单源 cross 比较框架 |
| 配置 | `configs/session_multisource_compare.yaml` | **Step 1 多源** ses-01+02→ses-03 |
| 数据/split | `src/data/session_splits.py` | `load_ok_sessions`、within StratifiedKFold、单源 cross 有向对、标签→{0,1}、JSON 持久化 |
| 漂移(A) | `src/analysis/session_drift.py` + `scripts/analysis/run_session_drift.py` | MMD/CORAL/μβ shift/KS/ERD-ERS/CSP/RMS/Fisher（matplotlib-only） |
| 模型(B) | `src/models/{eegnet,deepconvnet,fbcnet,cap_eegnet}.py` + `registry.py` | 4 模型统一 `{logits,features,confidence}` 契约 |
| 训练 | `src/training/trainer.py` | 通用 trainer（CE + 可选 confidence-BCE + 早停） |
| 协议(B/C) | `src/evaluation/session_protocols.py` | within CV + 单源 cross（无泄漏） |
| 协议(Step 1) | `src/evaluation/session_multisource_protocols.py` | **多源 ses-01+02→ses-03**（无泄漏，val 只从 train carve） |
| 入口 | `scripts/train_session_models.py`、`scripts/train_session_multisource.py` | within/单源 cross / 多源 训练入口 |
| 汇总 | `scripts/summarize_session_results.py`、`scripts/summarize_multisource_results.py` | 结果表/图/排名/报告 |
| 文档 | `docs/SESSION_DRIFT_ANALYSIS.md`、`docs/BASELINE_PROTOCOL.md`、`docs/MULTISOURCE_STEP1_REPORT.md` | 漂移指标说明 + 协议 + Step 1 报告 |
| 参考 | `docs/references/P10_MI_generalization/`、`docs/references/senior_scripts/` | 学长 P10 材料 + 原始脚本（只读参考，不在此运行） |

---

## 1.6 规范项目结构（current vs future/legacy）

**当前主线（cross-session DG）用到的规范文件：**

```text
configs/  paths.yaml · preprocess.yaml · session_drift.yaml ·
          session_model_compare.yaml · session_multisource_compare.yaml
src/
  analysis/     session_drift.py
  data/         session_splits.py · shu_dataset.py · splits.py · manifest.py
  models/       eegnet.py · deepconvnet.py · fbcnet.py · cap_eegnet.py · registry.py
  training/     trainer.py
  evaluation/   session_protocols.py · session_multisource_protocols.py · metrics.py · data_quality.py
  preprocessing/ neuracle_events.py · eog_ecg_clean.py · pipeline.py · shu_preprocess.py
  utils/        paths.py · config.py · io.py · logging_utils.py · seed.py
  visualization/ quality_plots.py
scripts/
  build_manifest.py · check_raw_bdf.py · preprocess_raw.py · preprocess_all.py · compare_processed_quality.py
  train_session_models.py · summarize_session_results.py · baseline_report.py
  train_session_multisource.py · summarize_multisource_results.py
  analysis/run_session_drift.py · analysis/build_drift_report.py
  slurm/  session_drift_cpu.sbatch · train_session_models_gpu.sbatch · summarize_session_results_cpu.sbatch ·
          train_session_multisource_gpu.sbatch · summarize_multisource_results_cpu.sbatch ·
          baseline_report_cpu.sbatch · preprocess_cpu.sbatch · compare_quality_cpu.sbatch
docs/   PROJECT_OVERVIEW.md · PROJECT_STATUS_CURRENT.md · EXPERIMENT_PROTOCOL.md · BASELINE_PROTOCOL.md ·
        SESSION_DRIFT_ANALYSIS.md · RESULTS_SUMMARY.md · MULTISOURCE_STEP1_REPORT.md ·
        NEXT_EXPERIMENT_PLAN.md · ADAPTATION_BASELINE_PLAN.md · CODE_INTEGRATION_NOTES.md ·
        P10_INTEGRATION_SUMMARY.md · PROGRESS.md · EXPERIMENT_LOG.md · references/
outputs/   analysis/session_drift_v1/ ·
           experiments/static_baseline_v1/{within_session,cross_session,figures,STATIC_BASELINE_REPORT.md} ·
           experiments/session_model_compare_v1/{runs,splits,summaries} ·
           experiments/session_multisource_v1/{runs,splits,summaries,figures}
checkpoints/ session_model_compare_v1/{eegnet,deepconvnet,fbcnet}/ ·
             session_multisource_v1/{eegnet,deepconvnet,fbcnet}/
```

**FUTURE / legacy（保留在仓库，当前不跑）：**
- 41/10 跨被试主线脚手架：`scripts/{train_cross_subject,finetune_target,run_online_adaptation,
  train_baseline,make_splits}.py`、`configs/{train_cross_subject,finetune,online_adaptation,eegnet_baseline}.yaml`、
  对应 `scripts/slurm/*_gpu.sbatch`、`splits/cap_eegnet_4110_seed*.json`、`tests/`、`scripts/sanity_train.py`。
- 愿景/路线文档：`docs/{ROADMAP,MODEL_PLAN,ALIGNMENT_CHECKLIST}.md`（full CAP-EEGNet 长期目标）。
- CAP-EEGNet 在 `src/models/cap_eegnet.py` 保留 **v1**，其余组件 `NotImplementedError`。

> ⚠️ git 注意：当前 HEAD 仍是 2026-06-04 的 scaffold commit，之后的所有工作（漂移/baseline/Step 1/文档）
> **尚未提交**。建议尽快提交，避免再次出现工作树丢失（详见 `docs/PROGRESS.md` 2026-06-09 条目）。

---

## 2. 各部分详解：做了什么、做成了什么

### 2.1 路径管理（核心机制）
- **`configs/paths.yaml`**：唯一的路径来源。含 `raw_data.shu_2c_root`（外部只读根
  `/share/workspace2/moto_imagination/WBCIC_SHU`）、`processed_data.eog_ecg_clean_root`
  （正式输出，外部 `.../WBCIC_SHU/processed/eog_ecg_clean`）、`manifests.processed_manifest`、`splits.dir`。
- **`src/utils/paths.py`**：`load_paths()` 读 yaml（环境变量 `SHU_2C_ROOT` 可覆盖），
  校验存在性，缺失则抛清晰错误。**任何脚本都不硬编码数据集路径。**

### 2.2 预处理（已完成全量 + QC）
- **`neuracle_events.py`**：解析 evt.bdf TAL 通道，提取 200 触发（MNE 直接读会漏）。
- **`eog_ecg_clean.py`（正式）**：按名识别 EEG/EOG/ECG → 校验辅助通道 → MNE ICA 去眼/心 →
  论文式后半（去辅助→重参考 Pz 去 Pz=58→0.5–40 带通+50 陷波→切 [0,4)s 全段去均值→250Hz）→
  `[200,58,1000]` float32(µV)。ICA 失败退化 no-aux-clean（记录不崩）。
- **全量结果**：153 session → 148 ok / 5 failed（trigger<200 的 5 个：sub-023/ses-01、
  sub-024/ses-02、sub-024/ses-03、sub-026/ses-01、sub-032/ses-02）。对比官方 derivatives QC PASS。

### 2.3 漂移诊断（A，已完成）
- `src/analysis/session_drift.py`：MMD/CORAL/μβ power shift/μ-KS/ERD-ERS 空间相关/CSP 相似度/
  RMS 比值/Fisher shift。144 pairs / 50 subjects。
- 核心结论：跨 session 漂移主要是**空间模式 + μ/β 频谱分布**（CSP≈0.420、ERD-μ≈0.419、μ-KS≈0.246），
  幅值稳定（RMS 中位数≈0.992），可分性无系统变化（Fisher≈0）。详见 `docs/SESSION_DRIFT_ANALYSIS.md`、
  `docs/RESULTS_SUMMARY.md`。

### 2.4 模型与协议（B / Step 1，已完成）
- 4 模型（EEGNet/DeepConvNet/FBCNet/CAP-EEGNet v1）共享 `{logits, features, confidence}` 契约 +
  统一 trainer + 统一指标，保证公平对比。
- **within-session CV**（每 session 10-fold，上界）与 **单源 cross-session**（同被试有向 session 对）：
  `src/evaluation/session_protocols.py`。
- **Step 1 多源 cross**（ses-01+02 合并训练，ses-03 测试，val 只从 train carve）：
  `src/evaluation/session_multisource_protocols.py`。
- 详细协议与无泄漏规则见 `docs/BASELINE_PROTOCOL.md`。

### 2.5 文档与规则
- 规则在 `.cursor/rules/`；`PROGRESS.md` 逐日记忆；`PROJECT_STATUS_CURRENT.md` 权威状态；
  `RESULTS_SUMMARY.md` 汇总结果；`MULTISOURCE_STEP1_REPORT.md` Step 1 深度分析；
  `NEXT_EXPERIMENT_PLAN.md` + `ADAPTATION_BASELINE_PLAN.md` 下一步计划。

---

## 3. 关键技术决策与验证结果（已核实）

| 项 | 结论 |
|---|---|
| 通道 | 64 = 59 EEG + **1 ECG(`ECG`) + 4 EOG(`HEOR/HEOL/VEOU/VEOL`)**；`eeg.json` 计数反，以通道名为准 |
| 参考/降维 | 重参考 Pz 后去 Pz → 58 EEG |
| 采样率 | raw 1000Hz → 250Hz；4s 窗 = 1000 点 |
| 标签 | 触发 1→0(左手)、2→1(右手)；每 session 100/100 |
| 事件 | 在 evt.bdf 的 TAL 通道，需自研解析器；MNE 直接读会漏 |
| 单位 | BDF 物理单位是乱码 `?V`(µV)，MNE 不换算 → 直接存 µV，**不要 ×1e6** |
| 单 session 验证 | sub-001/ses-01：`[200,58,1000]`、与论文 `.mat` 标签逐个一致、相关 0.994、std 11.28 vs 11.26 |
| GPU 环境 | `mi_torch_cu118`（torch 2.7.1+cu118，cuda 11.8），RTX 4090 D |
| Slurm | 分区 `gpu2node`(默认)/`gpu3node`，`gpu:8` |

---

## 4. 当前结果速览（已完成）

### 4.1 静态 baseline（5 seed，mean ± std）

| 模型 | within | 单源 cross | drop | **多源 ses-01+02→ses-03** |
|---|---:|---:|---:|---:|
| EEGNet | 0.807 | 0.711 | 11.9% | **0.7717 ± 0.003** |
| DeepConvNet | 0.766 | 0.681 | 11.1% | **0.7564 ± 0.007** |
| FBCNet | 0.720 | 0.628 | 12.8% | **0.6750 ± 0.002** |

> 多源对三个模型都优于最强单源 `ses-02→03`（EEGNet +0.0224 / DCN +0.0353 / FBC +0.0267），
> 详见 `docs/MULTISOURCE_STEP1_REPORT.md`。

### 4.2 与论文 within 的差距
论文 within 10-fold CV：EEGNet 85.32 / DeepConvNet 84.47 / FBCNet 78.40（%）；我们 80.67 / 76.63 / 72.03，
低 4.65 / 7.84 / 6.37 pp。诊断为**训练配方/数据预算**差异（within 折内再切 20% val），**非架构 bug**
（排序 + S1<S2<S3 趋势与论文一致），不影响 within-vs-cross 的核心结论。详见 `docs/RESULTS_SUMMARY.md` §4。

---

## 5. 环境准备（每次开终端先做）

```bash
source /share/software/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate mi_torch_cu118      # GPU 训练用 cu118 环境
cd /share/home/yuan/SYX/eeg-mi-online
```

> 规则：登录节点只做 < ~30s 的轻量操作。**预处理/训练/任何 GPU 任务都不能在登录节点跑**，
> 要么 `sbatch`，要么 `srun` 进计算节点。

---

## 6. 如何跑当前主线的实验（按阶段，标注状态）

### A 漂移诊断 ✅ 已完成（可复跑，CPU）

```bash
sbatch scripts/slurm/session_drift_cpu.sbatch          # 全量
# -> outputs/analysis/session_drift_v1/
```

### B within + 单源 cross baseline ✅ 已完成（GPU）

```bash
sbatch scripts/slurm/train_session_models_gpu.sbatch \
    --config configs/session_model_compare.yaml --models eegnet,deepconvnet,fbcnet --protocol both
sbatch scripts/slurm/summarize_session_results_cpu.sbatch
# -> outputs/experiments/session_model_compare_v1/summaries/
```

### Step 1 多源 cross（ses-01+02→ses-03）✅ 已完成（GPU）

```bash
# smoke（GPU 节点 srun，2 被试，eegnet，seed 0，3 epoch）
python scripts/train_session_multisource.py --config configs/session_multisource_compare.yaml \
    --models eegnet --subjects 1,2 --seeds 0 --max-epochs 3 --device cuda
# full（5 seed × 3 模型）
sbatch scripts/slurm/train_session_multisource_gpu.sbatch \
    --config configs/session_multisource_compare.yaml --models eegnet,deepconvnet,fbcnet --seeds 0,1,2,3,4
sbatch scripts/slurm/summarize_multisource_results_cpu.sbatch
# -> outputs/experiments/session_multisource_v1/summaries/
```

### Step 2 no-learning adaptation baseline 🔜 下一步（未实现/未跑）

见 `docs/ADAPTATION_BASELINE_PLAN.md`（none / session_zscore / EA / Riemannian / BN-stats / filter-bank）。

### 在线 / 41-10 / 微调 / CAP-EEGNet full 🚧 future（不跑）

只有 Step 2 结果出来后才讨论。

---

## 7. 数据流串讲

```text
外部 raw BDF (51被试×3session, 1000Hz, 64ch)
   │  build_manifest.py
   ▼
manifests/shu_2c_raw_manifest.csv
   │  preprocess_all.py (mode=eog_ecg_clean: TAL 触发 + ICA 去眼/心 + 论文式后半)
   ▼
<eog_ecg_clean_root>/sub-*/ses-*/*.npz  (X[200,58,1000] µV, y[200]∈{0,1}) + processed_manifest.csv
   │  仅 status=ok 的 148 session
   ├──【A 漂移诊断】被试内 session pair 的 MMD/CSP/ERD/KS/... → 诊断报告
   ├──【B 静态 baseline】within-session 10-fold CV（上界）+ 单源有向 cross（漂移代价）
   ├──【Step 1 多源】train=ses-01+ses-02，test=ses-03（val 只从 train carve）
   └──【Step 2 future】在 cross 协议上加 no-learning 对齐（EA/Riemannian/BN/filter-bank/zscore）
```

**铁律**：数据入口仅 status=ok 的 `.npz`，`derivatives/.mat` 只作标签对照；按被试/session 划分绝不按 trial 泄漏；
cross/多源/adaptation 的 val 只能从 train carve、绝不用 test session 的 label；路径不硬编码；
不写入外部 raw / workspace2 原数据目录；重任务只走 Slurm（GPU 用 `mi_torch_cu118`）。

---

## 8. 下一步顺序

1. ✅ A 漂移诊断；B within + 单源 cross baseline；Step 1 多源 ses-01+02→ses-03。
2. 🔜 **Step 2 no-learning adaptation baseline**（`docs/ADAPTATION_BASELINE_PLAN.md`）。
3. 🚧 Step 3+（future）：等 Step 2 结果再决定 online / adapter / prototype / memory / CAP-EEGNet full / 41-10 / 微调。

> 详细计划见 `docs/NEXT_EXPERIMENT_PLAN.md`；长期愿景见 `docs/ROADMAP.md`。

---

## 9. 命令速查表

```bash
# 环境
source /share/software/anaconda3/2024.10/etc/profile.d/conda.sh && conda activate mi_torch_cu118
cd /share/home/yuan/SYX/eeg-mi-online

# Slurm 监控
squeue -u $USER ;  scancel <jobid> ;  sacct -j <jobid> ;  tail -f logs/slurm/<name>-<jobid>.out
```
