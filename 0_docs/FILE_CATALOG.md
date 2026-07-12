---
title: "File Catalog"
tags:
  - "#modality/eeg"
created: "2026-06-10"
updated: "2026-07-12"
status: "active"
---

# File Catalog

新增 source/config/doc/结果文件时更新本表。

## 根目录

| 路径 | 作用 |
|:---|:---|
| `AGENTS.md` | 唯一权威灵魂记忆。 |
| `CLAUDE.md` | 兼容入口，指向 AGENTS.md（供 Claude Code 等原生读取）。 |
| `README.md` | 人类入口与结构总览。 |
| `PHASE3_ROUTE_PLAN.md` | Phase 3 路线 v2.1（Oracle 先裁决；学长已批 A0/A1）。 |
| `proposal.md` | 项目提案。 |
| `progress.md` | 进度日记（PROGRESS 角色，逐条追加）。 |
| `experiment_log.md` | 实验日志速查。 |
| `results.md` | 结果速查表。 |

## 0_docs

| 路径 | 作用 |
|:---|:---|
| `0_docs/ARCHITECTURE.md` | 结构 + 代码分层 + 文件职责。 |
| `0_docs/STATUS.md` | 进度 + 运行就绪 + SHU 就绪 + 下一步 + 清理策略。 |
| `0_docs/FILE_CATALOG.md` | 本文件，文件索引。 |
| `0_docs/operation_log.md` | 文件系统操作日志。 |

## 阶段结果

| 路径 | 作用 |
|:---|:---|
| `1_session_drift/{wbci_shu,shu}/{report,tables,figures}/` | Phase 0 漂移诊断结果（数据集并列）。 |
| `2_baseline/{wbci_shu,shu}/no_alignment_baseline/{report,tables,figures}/` | Phase 1 baseline + Phase 2a multi-source（两数据集 done；含 `report/AI_ANALYSIS.md`）。 |
| `2_baseline/{wbci_shu,shu}/alignment_baseline/{report,tables,figures}/` | Phase 2b alignment（两数据集 done；含 `report/AI_ANALYSIS.md`）。 |
| `4_experiments/{wbci_shu,shu}/prototype_drift/{report,tables,figures}/` | Phase 2c prototype drift（两数据集 done；含 `report/AI_ANALYSIS.md`）。 |
| `PHASE3_ROUTE_PLAN.md`（根目录） | **Phase 3 完整路线计划 v2.1（已批准 A0/A1）**：Oracle 先裁决 + 7 条硬约束。 |
| `3_online_adaptation/PHASE3_TTA_DESIGN.md` | **Phase 3 技术总纲**：T3A 方法/公式/矩阵/诚实警示 + 实现路线 v2。 |
| `3_online_adaptation/PRETRAINED_MODEL_INTEGRATION_CONTRACT.md` | **预训练模型接入权威契约**（交付物/能力矩阵/步骤/preflight）。 |
| `3_online_adaptation/` | **设计文档区**；正式实验结果不放这里。 |
| `4_experiments/{wbci_shu,shu}/tta/` | **Phase 3 TTA 正式结果区**（smoke / full A0 replay / oracle_diagnostic / method_catalog / reports）。 |
| `5_papers/{wbci_shu,shu}/` | 论文材料。 |
| 每层目录的 `README.md` | 数据集层/实验层/叶子层说明，由 `scripts/scaffold_readmes.py` 生成。 |

## 代码

| 路径 | 作用 |
|:---|:---|
| `code/run.py` | 统一入口：dry-run / 训练 / `--summarize`，进程内调度。 |
| `code/runners.py` | Phase 0/1/2a/2b/2c/**phase3_tta** 的进程内 runner。 |
| `code/tta/` | **Phase 3 model-agnostic TTA backend**（adapters/feature_sources/methods/oracle/eval/report）。 |
| `code/tta/README.md` | TTA 包概览 + 契约指针。 |
| `code/tta/method_catalog.yaml` | 方法候选清单（多数只登记不实现）。 |
| `code/experiments/session_tta.py` | Phase 3 编排：smoke / opt-in `full_a0_replay` / SHU smoke。 |
| `code/methods/t3a.py` | 薄 re-export → `code.tta.methods.MinimalT3AMethod`。 |
| `code/experiments/prototype_drift.py` | Phase 2c：source-only 训练 + 冻结 embedding 提取 + prototype + 漂移指标 + 泄漏断言。 |
| `code/experiments/prototype_drift_summarize.py` | Phase 2c 汇总：合并 per-run CSV → tables/figures/report/run_status。 |
| `code/configs/experiments/phase2c_prototype_drift.yaml` | Phase 2c 实验配置（WBCIC）。 |
| `code/configs/experiments/shu_phase{0_drift_diagnostic,1_baseline,2b_alignment,2c_prototype_drift}.yaml` | SHU 4 个实验配置（32ch/5 session，数据集作用域输出）。 |
| `code/configs/experiments/{phase3_tta,shu_phase3_tta}.yaml` | **Phase 3 TTA 配置**（默认 smoke 安全开关）。 |
| `code/configs/experiments/phase3_tta_full_a0.yaml` | **Opt-in** WBCIC full A0 replay（非默认）。 |
| `tests/tta/` | TTA 行为测试（含 mock live-inference fixtures；CPU）。 |
| `code/preprocessing/shu_mat.py` | SHU per-session `.mat` → 标准化 `.npz` 的核心加载/校验。 |
| `scripts/preprocess_shu.py` | SHU 全量预处理：`.mat` → npz + `processed_manifest.csv`。 |
| `scripts/build_drift_report.py` | session/dataset 无关的 drift per-pair/per-subject 表+图+报告构建器（泛化 legacy WBCIC-only 版）。 |
| `scripts/make_baseline_cross_all.py` | Phase 2b baseline schema 适配器：Phase 1 cross（accuracy/train_session）→ alignment 口径 `results_cross_session_all.csv`（acc/train_sessions/training_scope），数据集无关。 |
| `scripts/scaffold_readmes.py` | 双数据集结果树各层 README 生成器。 |
| `scripts/slurm/shu_gpu.sbatch` | SHU 通用 GPU 训练 Slurm 脚本（config + passthrough，mi_torch_cu118）。 |
| `scripts/slurm/shu_cpu.sbatch` | SHU 通用 CPU Slurm 脚本（Phase 0 drift / `--summarize`）。 |
| `checkpoints/README.md` | checkpoint 命名规范（dataset/run_id/method/model/任务前缀/sub/session/seed）。 |
| `scripts/slurm/train_prototype_drift_gpu.sbatch` | Phase 2c 单 (model,seed) GPU 训练 job。 |
| `scripts/slurm/summarize_prototype_drift_cpu.sbatch` | Phase 2c CPU 汇总 job（afterany 依赖）。 |
| `scripts/slurm/submit_prototype_drift_full.sh` | 提交 15 GPU + 1 summarizer 全量任务并记录 job ids。 |
| `code/summaries/session.py` | Phase 1 baseline 汇总（表/图/原始报告）。 |
| `code/summaries/multisource.py` | Phase 2a multi-source 汇总。 |
| `code/summaries/alignment.py` | Phase 2b alignment 汇总。 |
| `code/summaries/canonical.py` | 按 9 段结构从汇总 CSV 生成 canonical `REPORT.md`。 |
| `code/summaries/summarize.py` | `--summarize` 调度器：原始汇总 + canonical。 |
| `code/configs/datasets/*.yaml` | 数据集配置（wbci_shu / shu）。 |
| `code/configs/models/*.yaml` | 模型超参配置。 |
| `code/configs/experiments/*.yaml` | 阶段实验配置。 |
| `code/datasets/` | 数据集适配器与 split/通道映射。 |
| `code/models/` | 模型与 registry。 |
| `code/methods/` | 对齐与适应方法。 |
| `code/experiments/` | 漂移/baseline/multi-source/alignment 协议 + 指标。 |
| `code/training/` | 通用 trainer。 |
| `code/preprocessing/` | 预处理逻辑。 |
| `code/utils/` | config/io/logging/paths/seed。 |

## backup

| 路径 | 作用 |
|:---|:---|
| `backup/README.md` | backup 索引与策略。 |
| `backup/COMPLETED_ARTIFACTS_INDEX.md` | 历史结果/权重/日志归档位置。 |
| `backup/root_archive_2026-06-10/` | 清理前的旧代码/文档/产物/权重/日志。 |
| `backup/legacy_snapshot_2026-06-10/` | 更早的轻量快照。 |
