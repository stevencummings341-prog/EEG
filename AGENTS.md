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

## 1. Project Identity

| 维度 | 内容 |
|:---|:---|
| 项目 | 多数据集运动想象 EEG 跨 session 泛化研究 |
| 核心问题 | 跨 session MI-EEG 解码性能下降的根因是什么，如何修复？ |
| 数据集 | WBCIC-SHU 2025 + SHU 2022 |
| 任务 | 二分类运动想象：左手 vs 右手 |
| 当前主线 | Phase 2c Prototype Drift Analysis 准备阶段 |
| 长期方向 | adaptation -> prototype memory -> online/test-then-update -> agent/tool routing |

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

Phase 2c: Prototype Drift Analysis
  -> 当前下一步：验证跨 session 掉点是否来自 embedding / task prototype 漂移

Phase 3+: Adaptation / Memory / Online / Agent
  -> future；必须由前面实验结果支撑，不为了 Agent 而 Agent
```

**铁律**：每一步都必须有上一步结果支撑。未运行的内容只能写 future / planned，绝不能写 done。

## 3. Current Verified Facts

- WBCIC-SHU 2C processed entry:
  `/share/workspace2/moto_imagination/WBCIC_SHU/processed/eog_ecg_clean/processed_manifest.csv`
- WBCIC usable sessions: 148 ok / 5 failed.
- WBCIC tensor convention: `X [trials, 58, 1000]`, `y [trials]`, labels normalized to `{0,1}`.
- SHU 2022 source:
  `/share/workspace2/moto_imagination/SHU`
- SHU metadata: 25 subjects, 5 sessions, 32 EEG channels, 250 Hz, read-only external source.
- Completed result: no-learning alignment is insufficient; best BN-stat gain is small and below +2pp.
- Future work: Prototype Drift, prototype adaptation, memory, online, full CAP-EEGNet, 41/10 cross-subject pretraining.

## 4. Hard Scope

1. Write only inside `/share/home/yuan/SYX/eeg-mi-online` unless the user explicitly says otherwise.
2. Treat `/share/home/yuan/SYX/P10_MI泛化研究` as read-only reference.
3. Treat `/share/home/yuan/SYX/CLAUDE.md` as read-only upstream reference.
4. Treat `/share/workspace2/moto_imagination/WBCIC_SHU` and `/share/workspace2/moto_imagination/SHU` as read-only data sources.
5. Do not move, rename, delete, or create files under `/share/workspace2/...`.
6. Do not overwrite completed `*_v1` outputs/checkpoints.

## 5. Directory Routing

| 目录 | 内容 | 规则 |
|:---|:---|:---|
| `0_docs/` | 文档中心：ARCHITECTURE / STATUS / FILE_CATALOG / operation_log | 新文档必须同步 `FILE_CATALOG.md` |
| `1_session_drift/` | Phase 0 漂移诊断真实结果（报告+CSV+图） | 已完成结果只读，复跑必须新 run_id |
| `2_baseline/` | Phase 1 baseline + 2a multi-source + 2b alignment 真实结果 | 已完成结果只读，禁止覆盖 |
| `3_online_adaptation/` | online/adaptation 设计区 | future，不写成已验证 |
| `4_experiments/` | Phase 2c+ 新实验入口 | 每个实验一个子目录 + README |
| `5_papers/` | 论文、PPT、图表材料 | 不放 raw EEG / checkpoint |
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
- Phase 0/1/2a/2b runners are implemented in `code/runners.py`; summarizers + canonical 9-section report in `code/summaries/`.
- Train: `python code/run.py --config code/configs/experiments/<phase>.yaml [...]`.
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
