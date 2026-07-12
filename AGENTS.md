# AGENTS.md — Project Soul Memory

> 唯一权威灵魂记忆文件。后续 Agent / 人类进入本项目，先读本文件。  
> `CLAUDE.md` 只是兼容入口，指向本文件；`.cursor/rules/` 是机器可读规则切片，不再作为独立灵魂文件。

## 0. Authoritative Files

| 类型 | 文件 | 作用 |
|:---|:---|:---|
| 灵魂记忆 | `AGENTS.md` | 本文件，唯一权威项目记忆与执行准则。 |
| 进度日记 | `progress.md` | PROGRESS 角色，逐条追加运行记忆，最新在上。 |
| 架构+结构 | `0_docs/ARCHITECTURE.md` | 目录结构、代码分层、每个文件作用。 |
| 状态+就绪 | `0_docs/STATUS.md` | 进度、能否跑、SHU 就绪、下一步、清理策略。 |
| 文件索引 | `0_docs/FILE_CATALOG.md` | 新增文件后必须同步。 |
| 操作日志 | `0_docs/operation_log.md` | 只记录创建、删除、移动、重命名。 |

> **核心灵魂集（交接 / ChatGPT 导入用）= `AGENTS.md` + `progress.md` + `0_docs/STATUS.md` + `0_docs/ARCHITECTURE.md`。**
> 用户会定期把这 4 份导入 ChatGPT 做项目背景更新；**Agent 每次有实质变动必须立刻同步更新它们（见 §8.5），不要让知识只留在聊天里。** 跨工具/跨机器（Cursor SSH、Claude Code、ChatGPT）一律以 `AGENTS.md` 为准；`CLAUDE.md` 是 Claude Code 的自动入口（内容指向本文件）。这些都随项目文件夹走，任何机器 SSH 到本目录即自动生效（`.cursor/rules/` 与 `AGENTS.md` 都在本文件夹内）。

## 1. Project Identity

| 维度 | 内容 |
|:---|:---|
| 项目 | 多数据集运动想象 EEG 跨 session 泛化研究 |
| 核心问题 | 跨 session MI-EEG 解码性能下降的根因是什么，如何修复？ |
| 数据集 | WBCIC-SHU 2025 + SHU 2022 |
| 任务 | 二分类运动想象：左手 vs 右手 |
| 当前主线 | **Phase 3 — 跨 session 修复（Oracle 先裁决，再看 T3A）**（planned；路线见根目录 `PHASE3_ROUTE_PLAN.md` + `3_online_adaptation/PHASE3_TTA_DESIGN.md`）。两数据集 Phase 0–2c 均已完成（诊断收官），从诊断转向修复 |
| SHU 并列线 | SHU 2022 **Phase 0/1/2a/2b/2c 全部完成**（2026-07-06 summarize + AI 分析补齐，与 WBCIC 齐平）。结论同质：无学习对齐不足、scatter 膨胀 / Fisher collapse、cosine>euclidean、FBCNet 弱；差异：SHU 近 chance（地板效应）、最佳对齐是 z-score 而非 BN-stats、prototype 信号更噪 |
| 当前第一优先（2026-07-12，**pretrained-model-ready 工程态**） | **等待学长真实预训练模型接入**。已完成：Round-1 scaffold + mock live-inference 验证 + **WBCIC full A0 complete（4320/4320, max\|Δ\|=0）** + SHU no_tta replay smoke + 交接契约 `3_online_adaptation/PRETRAINED_MODEL_INTEGRATION_CONTRACT.md`。**不是** full T3A / 正式 Oracle 裁决；**真实预训练模型尚未接入**。下一步：新 adapter + checkpoint → preflight → live smoke → 再 Phase 3B Oracle。硬约束见 `PHASE3_ROUTE_PLAN.md` §2.5。 |
| 长期方向 | Oracle 裁决 → (safe/reliability-weighted) T3A → prototype memory → online/test-then-update → agent/tool routing；Tent/SHOT/CoTTA 为梯度类对照；集成真实 EEG 基础模型 (CBraMod/REVE) 跑 TTA 为可选延伸 |

> **架构升级（2026-06-11）**：所有结果区 + outputs + checkpoints 改为「数据集并列」：
> `<结果区>/{wbci_shu,shu}/...`、`outputs/.../{wbci_shu,shu}/...`、`checkpoints/{wbci_shu,shu}/...`。
> 每一层目录都有 README。checkpoint 命名规范见 `checkpoints/README.md`。

## 2. Research Logic Chain

```text
Phase 0: Drift Diagnostic
  -> 发现漂移主要是空间模式 + μ/β 频谱分布，不是幅值

Phase 1: Baseline
  -> 发现跨 session drop 约 10pp

Phase 2a: Multi-source Baseline
  -> 多源 ses-01+02 -> ses-03 优于最强单源

Phase 2b: No-learning Alignment Baseline
  -> 统计对齐不够；BN-stats 只有小幅正收益，没人超过 +2pp

Phase 2c: Prototype Drift Analysis（done, 两数据集）
  -> 全量跑完（WBCIC 4320 / SHU 7500 cells 全 ok）。prototype drift 显著但中等（多元 R²≈0.35）；
     机制 = within-class scatter 膨胀 / Fisher collapse（4.58→1.57），非 centroid collapse；
     cosine 几何优于 euclidean；EEGNet/DeepConvNet 清晰、FBCNet 弱。
  -> 判定 qualified go：转向基于原型的测试时适应。

Phase 3: 跨 session 修复 —— Oracle 先裁决，再看 T3A（planned — 当前主线，路线 v2）
  -> 依据 Phase 2c 权威结论：prototype drift 只解释部分掉点，主机制是 within-class scatter 膨胀 /
     Fisher collapse。故不能直接大规模上 T3A，必须先做 Oracle 上限裁决。
  -> A0 replay + No-TTA 精确复现（|Δ|<1e-6）-> A1 minimal T3A smoke（只证 pipeline）
     -> B ⭐Oracle 上限裁决门⭐（source_proto/target_label/cosine/mahalanobis/shrinkage/reliability-weighted）
        · Oracle >+3pp -> C T3A ablation -> D safe-T3A -> E 全量 -> (F Tent/SHOT/CoTTA 对照) -> G 论文叙事
        · Oracle <+1pp -> 停大规模 T3A，转 scatter/reliability/decision-boundary 机制（论文路线 3）
  -> 参考 arXiv:2604.16926v1 (NeuroAdapt-Bench)：Tent/SHOT/T3A 中只有 T3A(无优化、原型+熵筛选)最稳/唯一
     平均正增益，梯度类常负迁移。T3A = "用 target 特征重估 class prototype 无标签调整分类几何"。
  -> 复用 Phase 2c 已存嵌入，Oracle + T3A 全程离线 CPU 可跑（无需重训/GPU）。
  -> 铁律：backbone 是自训小模型非论文基础模型；跨 session 非跨数据集；二分类熵≡最大置信度(退化)，
     真正 ablation 轴是 filter_k/几何/软硬/margin；T3A 只作 baseline，未裁决前不写"最终修复方案"。
     详见根目录 PHASE3_ROUTE_PLAN.md 与 3_online_adaptation/PHASE3_TTA_DESIGN.md。

Phase 4+: Memory / Online test-then-update / Agent
  -> future；必须由前面实验结果支撑，不为了 Agent 而 Agent
```

**铁律**：每一步都必须有上一步结果支撑。未运行的内容只能写 future / planned，绝不能写 done。

## 3. Current Verified Facts

- WBCIC-SHU 2C processed entry:
  `/share/workspace2/moto_imagination/WBCIC_SHU/processed/eog_ecg_clean/processed_manifest.csv`
- WBCIC usable sessions: 148 ok / 5 failed.
- WBCIC tensor convention: `X [trials, 58, 1000]`, `y [trials]`, labels normalized to `{0,1}`.
- SHU 2022 raw source: `/share/workspace2/moto_imagination/SHU`（read-only，.edf/.mat/.tsv）。
- **SHU 2022 processed entry（已完成）**：
  `/share/workspace2/moto_imagination/SHU/processed/npz_clean/processed_manifest.csv`
  - 入口=作者发布的 per-session `.mat`（已带通/陷波/4s 切段），仅做标签归一化 {1,2}->{0,1} 转存 `.npz`，不再二次预处理。
  - 125 session 全 ok / 25 subjects。tensor: `X [trials, 32, 1000]`, `y [trials]∈{0,1}`。
  - 生成脚本：`scripts/preprocess_shu.py`（核心 `code/preprocessing/shu_mat.py`）。
- runner 数据集无关：manifest 由 config `data.manifest` 解耦（`code/runners.py:_resolve_manifest`），WBCIC 缺省回退 paths.yaml。
- 输出/权重按数据集作用域：`outputs/.../{wbci_shu,shu}/<run_id>/`、`checkpoints/{wbci_shu,shu}/<run_id>/`。
- Completed result: no-learning alignment is insufficient; best BN-stat gain is small and below +2pp.
- **SHU 结果（2026-07-06 全部完成）**：P1 within/cross(5-seed) EEGNet 0.611/0.538、DeepConvNet 0.606/0.536、FBCNet 0.553/0.508（cross 近 chance）；P2a ses01+02→03 EEGNet 0.544/DeepConvNet 0.558/FBCNet 0.512；P2b 无对齐 0.5274、最佳 session_zscore +1.42pp、无方法过 +2pp；P2c scatter 15.7→38.3(+144%)/Fisher 1.96→0.79(−60%)、fisher_change ρ=0.43、cosine>euclidean、FBCNet 弱。可读区 `2_baseline/shu/{no_alignment_baseline,alignment_baseline}/`、`4_experiments/shu/prototype_drift/`，各含 AI_ANALYSIS.md。
- **2026-07-06 修复**：① `prototype_drift_summarize.summarize_from_cfg` 读 `data.manifest`（原 `data.manifest_path`，SHU 会误用 WBCIC 网格）；② 新增 `scripts/make_baseline_cross_all.py` 适配 Phase 2b `baseline_cross_all` 的 schema（accuracy/train_session → acc/train_sessions + training_scope）。
- **Phase 3 route（planned，当前主线，v2 = Oracle 先裁决）**：跨 session 修复。权威路线文件
  = 根目录 `PHASE3_ROUTE_PLAN.md`（待审批的完整路线）+ `3_online_adaptation/PHASE3_TTA_DESIGN.md`（技术总纲）。
  依据 Phase 2c（主机制=scatter 膨胀，非 centroid collapse），**Oracle 上限裁决门提前到 T3A 大扫之前**：
  A0 replay+No-TTA 精确复现 → A1 minimal T3A smoke → **B Oracle 裁决（>+3pp 才扩大 T3A，<+1pp 转 scatter/reliability）**
  → C ablation → D safe-T3A → E 全量 → F Tent/SHOT/CoTTA → G 叙事。参考 arXiv:2604.16926v1（NeuroAdapt-Bench，
  T3A 最稳/唯一平均正增益）。复用 Phase 2c 已存嵌入，离线 CPU 可跑。config 骨架
  `code/configs/experiments/{phase3_tta,shu_phase3_tta}.yaml`（runner `phase3_tta` 待实现 = Phase A0）。
  **⚠ 已知 bug**：WBCIC `embed_index__*.csv` 的 `npz_path` 列是失效旧路径（缺 `wbci_shu/`），Phase 3 runner
  须用 config `source_embeddings.embeddings_dir` 重拼路径，不信任该列。
- Future work: prototype memory, online test-then-update, full CAP-EEGNet, 41/10 cross-subject
  pretraining, 集成真实 EEG 基础模型 (CBraMod/REVE/TFM-Tokenizer) 跑 TTA。

## 4. Hard Scope

1. Write only inside `/share/home/yuan/SYX/eeg-mi-online` unless the user explicitly says otherwise.
   现在 `/share/home/yuan/SYX` 下只有本项目 `eeg-mi-online/` + `backups/` + 无关的 `run_test.sh`
   （另一个项目 AADSurvey 的 Slurm 模板，与本项目无关，不要用它）。旧的 `P10_MI泛化研究/` 与根目录
   `CLAUDE.md` 已被删除，不再是参考源——本项目自身即唯一权威。
2. Treat the RAW sources `/share/workspace2/moto_imagination/WBCIC_SHU`（sourcedata/derivatives）和 `/share/workspace2/moto_imagination/SHU`（.edf/.mat/.tsv）as read-only。禁止改名/删除/写入 raw。
3. 唯一允许写 workspace2 的位置是各数据集的 `processed/` 子树（预处理 npz+manifest 输出）：
   `WBCIC_SHU/processed/eog_ecg_clean/`、`SHU/processed/npz_clean/`。其余 workspace2 路径只读。
4. Do not overwrite completed `*_v1` outputs/checkpoints.

## 5. Directory Routing

| 目录 | 内容 | 规则 |
|:---|:---|:---|
| `0_docs/` | 文档中心：ARCHITECTURE / STATUS / FILE_CATALOG / operation_log | 新文档必须同步 `FILE_CATALOG.md` |
| `1_session_drift/{wbci_shu,shu}/` | Phase 0 漂移诊断真实结果（报告+CSV+图） | 数据集并列；已完成结果只读，复跑必须新 run_id |
| `2_baseline/{wbci_shu,shu}/` | Phase 1 baseline + 2a multi-source + 2b alignment 真实结果 | 下分 no_alignment_baseline/ + alignment_baseline/；只读，禁止覆盖 |
| `3_online_adaptation/{wbci_shu,shu}/` | online/adaptation 设计区 | future，不写成已验证 |
| `4_experiments/{wbci_shu,shu}/` | Phase 2c+ 新实验入口 | 每个实验一个子目录 + README |
| `5_papers/{wbci_shu,shu}/` | 论文、PPT、图表材料 | 不放 raw EEG / checkpoint |

> 上述每个结果区都是 `wbci_shu/` 与 `shu/` 并列，且每一层目录（含 report/tables/figures）都有 README。
> README 骨架可用 `scripts/scaffold_readmes.py` 重新生成。
| `code/` | 代码框架 | 人工入口只用 `code/run.py` |
| `inbox/` | 临时交接材料 | 读完后归档并更新索引 |
| `backup/root_archive_2026-06-10/` | 清理前的旧代码/文档/产物/权重/日志 | 保留追溯；根目录不再放旧层；阶段目录的可读结果即从这里复制 |
| `backup/legacy_snapshot_2026-06-10/` | 更早的轻量快照 | 历史追溯 |

## 6. Code Architecture Rule

Human-facing entry is:

```bash
python code/run.py --config code/configs/experiments/<phase>.yaml
```

Layering:

- Add dataset -> `code/datasets/<name>.py` + `code/configs/datasets/<name>.yaml`.
- Add model -> `code/models/<name>.py` + `code/configs/models/<name>.yaml`.
- Add method -> `code/methods/<name>.py`.
- Add experiment -> `code/experiments/<name>.py` + `code/configs/experiments/<phase>.yaml`.
- Every new file -> update nearest `README.md` + `0_docs/FILE_CATALOG.md`.

Current run capability:

- `code/run.py` is the new entry and is fully runnable in-process via `code/runners.py` (no dependency on the archived `scripts/`).
- Phase 0/1/2a/2b/2c runners are implemented in `code/runners.py`; summarizers + canonical 9-section report in `code/summaries/`.
- Train: `python code/run.py --config code/configs/experiments/<phase>.yaml [...]`.
- SHU configs（同一批 runner，仅数据/路径不同）：`code/configs/experiments/shu_phase{0_drift_diagnostic,1_baseline,2b_alignment,2c_prototype_drift}.yaml`。
  例：`python code/run.py --config code/configs/experiments/shu_phase1_baseline.yaml --device cuda`。
- 切换数据集只改 config（`data.name` + `data.manifest` + 输出路径前缀），不改 runner 代码。
- checkpoint 命名规范（数据集/实验/方法/模型/任务前缀/被试/session/seed）见 `checkpoints/README.md`。
- Summarize: `python code/run.py --summarize --config code/configs/experiments/<phase>.yaml` → tables/figures/native report + canonical `REPORT.md`.
- Verified: Phase 0 drift + Phase 1 EEGNet training (CPU smoke); Phase 2b summarize on archived 30150-row run CSVs (canonical report numbers match history).
- Old `src/scripts/configs` remain archived in `backup/root_archive_2026-06-10/` for reference only.
- Canonical reports MUST follow the 9-section structure in §8 (Core conclusion → Goal → Method → Protocol → Results → Analysis → Relationship to previous phases → Next step → File list).

## 6.5 Reporting Workflow (two layers)

There are two report layers; do not confuse them.

1. **Scripted report (no AI, automatable).** `python code/run.py --summarize --config <phase>` produces tables/figures, the native detailed report, and the canonical 9-section `REPORT.md`. This is deterministic and may be auto-run after training (e.g. a Slurm dependency job `--dependency=afterany:<train_ids>`). No AI and no manual narration needed.

2. **AI analysis report (manual trigger, per user's choice).** The user does NOT pre-write analysis and does NOT need to run anything special. After training+summarize have produced the data, the user opens Cursor/ChatGPT and says one line like “<phase> 跑完了，读结果写分析报告”. The agent then:
   - reads the phase's `tables/` CSVs + canonical `REPORT.md` (never fabricates numbers),
   - writes a deep analysis following the §8 9-section structure,
   - saves it to the phase's `report/` folder as `AI_ANALYSIS.md` (e.g. `2_baseline/no_alignment_baseline/report/AI_ANALYSIS.md`),
   - updates `progress.md` + `0_docs/operation_log.md`.

   Trigger phrases an agent should treat as "write the AI analysis report": “写分析报告 / 写 AI 分析 / 解读结果 / analyze results / write analysis report”. The AI report is grounded in the scripted tables; it adds cross-phase reasoning, anomalies, hypotheses, and next-step research direction that the templated report cannot.

## 7. Experiment Rules

1. Split by subject/session, never leak by trial.
2. Cross-session train/val come only from source/train data.
3. Target labels are only for final evaluation.
4. If Prototype Drift uses target labels for offline diagnostics, report must state: target labels are used only for offline diagnostic analysis, not for training or adaptation.
5. All experiments must record seed, data version, model hyperparameters, training recipe, hardware.
6. Heavy/GPU work must use Slurm and CUDA env `mi_torch_cu118`.
7. Smoke test before full run.
8. Mark a stage done only after output files exist on disk.

## 8. File Generation Rules

Markdown files should use YAML frontmatter:

```yaml
---
title: "File Title"
tags:
  - "#modality/eeg"
created: "YYYY-MM-DD"
updated: "YYYY-MM-DD"
status: "active"
---
```

Naming:

- No spaces in filenames.
- Prefer lowercase snake_case for logs/tables.
- Reports can use `*_REPORT.md` or descriptive `.md`.
- CSV headers use English lowercase snake_case.

Report structure:

1. Core conclusion first.
2. Goal.
3. Method.
4. Protocol.
5. Results.
6. Analysis.
7. Relationship to previous phases.
8. Next step.
9. File list.

## 8.5 Mandatory Handoff Updates

Every meaningful action must update the handoff memory immediately, not later:

1. If project structure changes, update `0_docs/ARCHITECTURE.md`.
2. If run readiness, cleanup policy, backup location, or deletable-file guidance changes, update `0_docs/STATUS.md`.
3. If a new file is created, update `0_docs/FILE_CATALOG.md` and the nearest `README.md`.
4. If files/directories are created, moved, renamed, or deleted, update `0_docs/operation_log.md`.
5. If experiment status, completed results, or next-step decisions change, update `progress.md`, `experiment_log.md`, and `results.md`.
6. If the change affects what another ChatGPT/Cursor agent must know, update this `AGENTS.md`.

This is required for cross-agent handoff. Do not leave structure/progress knowledge only in chat.

## 9. Historical Tracking and Backup Policy

Current state:

- Historical experiments, Slurm scripts, completed results, checkpoints, logs, and old compatibility layers have been moved into `backup/root_archive_2026-06-10/`.
- A lighter pre-cleanup snapshot remains in `backup/legacy_snapshot_2026-06-10/`.

Recommended future strategy:

- Do not delete `backup/root_archive_2026-06-10/`; it contains the old runnable/project history.
- Do not duplicate large checkpoints again.
- If full training is needed before direct `code/` runners are implemented, temporarily restore compatibility dirs from backup or implement direct runners first.

## 10. What Not To Do

- Do not write to `/share/workspace2`.
- Do not fabricate results.
- Do not silently run heavy jobs on login node.
- Do not collapse future work into current status.
- Do not move old compatibility layers into backup without updating paths and confirming risk.
- Do not delete checkpoints/results/logs just because they are large.
