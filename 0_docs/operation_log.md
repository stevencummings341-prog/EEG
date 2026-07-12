---
title: "Operation Log"
tags:
  - "#modality/eeg"
created: "2026-06-10"
updated: "2026-07-06"
status: "active"
---

# Operation Log

| Time | Action | Path | Note |
|:---|:---|:---|:---|
| 2026-07-12 | CREATE | `3_online_adaptation/PRETRAINED_MODEL_INTEGRATION_CONTRACT.md` | Pretrained-model integration contract (authoritative). |
| 2026-07-12 | CREATE | `code/configs/experiments/phase3_tta_full_a0.yaml` | Opt-in WBCIC full A0 replay config. |
| 2026-07-12 | CREATE | `code/tta/README.md` | TTA package overview. |
| 2026-07-12 | CREATE | `tests/tta/support/` | Test-only mock pretrained model + adapters (not production). |
| 2026-07-12 | CREATE | `tests/tta/test_live_model_inference.py` etc. | Live inference / capability / label-free interface tests. |
| 2026-07-12 | UPDATE | `code/tta/feature_sources/model_inference.py` | Real checkpoint→forward→FeatureBundle (was stub). |
| 2026-07-12 | UPDATE | `code/tta/adapters/base.py` | AdapterCapabilities + require_capability. |
| 2026-07-12 | UPDATE | `code/tta/oracle/label_guard.py` | Interface-level label strip in run_label_free. |
| 2026-07-12 | UPDATE | `code/experiments/session_tta.py` | full_a0_replay mode + dry_run no-I/O + run_t3a flag. |
| 2026-07-12 | CREATE | `4_experiments/wbci_shu/tta/reports/FULL_A0_REPLAY_VALIDATION_REPORT.md` | Full A0 complete evidence. |
| 2026-07-12 | CREATE | `4_experiments/wbci_shu/tta/replay_validation/full_a0_*.csv` | Full A0 tables. |
| 2026-07-12 | UPDATE | `4_experiments/shu/tta/{smoke,replay_validation}/` | SHU no_tta replay smoke artifacts. |
| 2026-07-10 | CREATE | `code/tta/` | Phase 3 Round-1 model-agnostic TTA backend scaffold. |
| 2026-07-10 | CREATE | `code/experiments/session_tta.py` | Phase 3 smoke/replay/oracle orchestration. |
| 2026-07-10 | CREATE | `code/methods/t3a.py` | Thin re-export of MinimalT3AMethod. |
| 2026-07-10 | CREATE | `tests/tta/` | 14 CPU unit/smoke tests. |
| 2026-07-10 | CREATE | `4_experiments/{wbci_shu,shu}/tta/` | Formal TTA result dirs + method_catalog + READMEs. |
| 2026-07-10 | UPDATE | `code/runners.py` | Register `phase3_tta`. |
| 2026-07-10 | UPDATE | `code/configs/experiments/{phase3_tta,shu_phase3_tta}.yaml` | Add round1 smoke safety switches. |
| 2026-07-10 | CREATE | `outputs/experiments/wbci_shu/tta_v1/` | Round-1 heavy outputs (new run_id, no overwrite of Phase 2c). |
| 2026-06-10 22:08:13 | CREATE | `0_docs/` | Project documentation hub. |
| 2026-06-10 22:08:13 | CREATE | `1_session_drift/` | Phase 0 result index. |
| 2026-06-10 22:08:13 | CREATE | `2_baseline/` | Baseline/alignment result index. |
| 2026-06-10 22:08:13 | CREATE | `3_online_adaptation/` | Future online design area. |
| 2026-06-10 22:08:13 | CREATE | `4_experiments/` | New experiment result area. |
| 2026-06-10 22:08:13 | CREATE | `5_papers/` | Paper/presentation area. |
| 2026-06-10 22:08:13 | CREATE | `code/` | Modular code framework. |
| 2026-06-10 22:08:13 | CREATE | `inbox/` | Temporary handoff material area. |
| 2026-06-10 22:08:13 | CREATE | `01_Lab_Journal/` | Filesystem operation log area. |
| 2026-06-10 22:12:36 | CREATE | `code/configs/datasets/README.md` | Add directory README. |
| 2026-06-10 22:12:36 | CREATE | `code/configs/models/README.md` | Add directory README. |
| 2026-06-10 22:12:36 | CREATE | `code/configs/experiments/README.md` | Add directory README. |
| 2026-06-10 22:12:36 | CREATE | `code/training/README.md` | Add directory README. |
| 2026-06-10 22:12:36 | CREATE | `code/preprocessing/README.md` | Add directory README. |
| 2026-06-10 22:12:36 | CREATE | `code/visualization/README.md` | Add directory README. |
| 2026-06-10 22:12:36 | CREATE | `code/online/README.md` | Add directory README. |
| 2026-06-10 22:19:00 | CREATE | `0_docs/STRUCTURE_AND_FILE_GUIDE.md` | Add detailed structure and file responsibility guide. |
| 2026-06-10 22:28:00 | UPDATE | `AGENTS.md` / `CLAUDE.md` | Consolidate project soul memory into `AGENTS.md`; keep `CLAUDE.md` as compatibility pointer. |
| 2026-06-10 22:40:31 | CREATE | `backup/` | Create backup index folder. |
| 2026-06-10 22:40:31 | CREATE | `backup/legacy_snapshot_2026-06-10/` | Copy lightweight legacy layers for traceability. |
| 2026-06-10 22:40:31 | CREATE | `backup/COMPLETED_ARTIFACTS_INDEX.md` | Index completed outputs/checkpoints/logs without moving large artifacts. |
| 2026-06-10 22:52:43 | MOVE | `01_Lab_Journal/` -> `0_docs/01_Lab_Journal/` | Move operation log out of root. |
| 2026-06-10 22:52:43 | MOVE | `src scripts configs docs tests manifests splits` -> `backup/root_archive_2026-06-10/` | Archive legacy compatibility layers. |
| 2026-06-10 22:52:43 | MOVE | `outputs checkpoints logs` -> `backup/root_archive_2026-06-10/` | Archive completed artifacts without deletion. |
| 2026-06-10 22:52:43 | MOVE | `requirements.txt` -> `code/requirements.txt` | Keep dependency file with code layer. |
| 2026-06-10 23:12:00 | COPY | `backup/.../session_drift_v1` -> `1_session_drift/` | Sync Phase 0 readable results into new architecture. |
| 2026-06-10 23:12:00 | COPY | `backup/.../baseline_v1` + `alignment_baseline_v1` -> `2_baseline/` | Sync Phase 1/2a/2b readable results into new architecture. |
| 2026-06-10 23:12:00 | RESTORE | `backup/.../docs/PROGRESS.md` -> `progress.md` | Restore running progress journal to root. |
| 2026-06-10 23:12:00 | MOVE | `0_docs/01_Lab_Journal/operation_log.md` -> `0_docs/operation_log.md` | Remove odd nested folder; flatten log location. |
| 2026-06-10 23:12:00 | MERGE | `STRUCTURE_AND_FILE_GUIDE + CODE_ARCHITECTURE` -> `0_docs/ARCHITECTURE.md` | Consolidate docs; delete originals. |
| 2026-06-10 23:12:00 | MERGE | `PROJECT_STATUS_CURRENT + RUN_READINESS_AND_CLEANUP_GUIDE` -> `0_docs/STATUS.md` | Consolidate docs; delete originals. |
| 2026-06-10 23:24:00 | REORG | `1_session_drift/` | Split into report/ tables/ figures/. |
| 2026-06-10 23:24:00 | REORG | `2_baseline/` | Split into no_alignment_baseline/ + alignment_baseline/, each report/ tables/ figures/. |
| 2026-06-10 23:57:00 | CREATE | `code/runners.py` | In-process Phase 0/1/2a/2b runners using code/ modules. |
| 2026-06-10 23:57:00 | UPDATE | `code/run.py` | Dispatch directly to code/runners (remove archived-script routing). |
| 2026-06-11 00:20:00 | CREATE | `code/summaries/{session,multisource,alignment,canonical,summarize}.py` | Bring summarizers into code/ + canonical 9-section report. |
| 2026-06-11 00:20:00 | UPDATE | `code/run.py` | Add --summarize action. |
| 2026-06-11 00:24:00 | UPDATE | `AGENTS.md` | Define two-layer reporting workflow: scripted summarize (auto) + manually-triggered AI analysis report. |
| 2026-06-11 01:30:00 | CREATE | `code/configs/experiments/phase2c_prototype_drift.yaml` | Phase 2c config (3 models, 5 seeds, 6 directed pairs, prototype/distance grids). |
| 2026-06-11 01:30:00 | CREATE | `code/experiments/prototype_drift.py` | Phase 2c experiment: source-only train + frozen embedding + prototype + drift metrics + leakage asserts. |
| 2026-06-11 01:30:00 | CREATE | `code/experiments/prototype_drift_summarize.py` | Phase 2c summarizer: merge per-run CSV -> tables/figures/report/run_status. |
| 2026-06-11 01:30:00 | UPDATE | `code/runners.py` / `code/summaries/summarize.py` | Register phase2c runner + summarize dispatch. |
| 2026-06-11 01:30:00 | CREATE | `scripts/slurm/{train_prototype_drift_gpu,summarize_prototype_drift_cpu}.sbatch`, `submit_prototype_drift_full.sh` | Phase 2c Slurm scripts (new architecture scripts/slurm dir). |
| 2026-06-11 01:50:00 | CREATE | `4_experiments/prototype_drift/` + `README.md` | Phase 2c readable result index (report/tables/figures synced by summarizer). |
| 2026-06-11 01:55:00 | SUBMIT | `outputs/experiments/prototype_drift_v1/` | Submit 15 GPU + 1 summarizer Slurm jobs (job ids 21536-21551); RUN_PLAN.md + full_job_ids.txt written. |
| 2026-06-11 18:35:00 | CREATE | `4_experiments/prototype_drift/report/AI_ANALYSIS.md` | AI deep analysis of completed Phase 2c results (grounded in tables). |
| 2026-06-11 20:05:00 | MOVE | `1_session_drift/{figures,report,tables}` -> `1_session_drift/wbci_shu/` | 结果区改双数据集并列：现有 WBCIC 结果移入 wbci_shu/。 |
| 2026-06-11 20:05:00 | MOVE | `2_baseline/{alignment_baseline,no_alignment_baseline}` -> `2_baseline/wbci_shu/` | 同上。 |
| 2026-06-11 20:05:00 | MOVE | `4_experiments/prototype_drift` -> `4_experiments/wbci_shu/prototype_drift` | 同上。 |
| 2026-06-11 20:05:00 | CREATE | `{1_session_drift,2_baseline,3_online_adaptation,4_experiments,5_papers}/shu/...` | 新建 SHU 并列空骨架（report/tables/figures）。 |
| 2026-06-11 20:05:00 | CREATE | 各级 `README.md`（数据集层/实验层/叶子层） | 每一层目录补 README，命名规范统一。 |
| 2026-06-11 20:06:00 | CREATE | `scripts/scaffold_readmes.py` | 双数据集结构 README 生成器（可复用）。 |
| 2026-06-11 20:10:00 | CREATE | `code/preprocessing/shu_mat.py`, `scripts/preprocess_shu.py` | SHU per-session .mat -> 统一 .npz + manifest 预处理链路。 |
| 2026-06-11 20:21:00 | CREATE | `/share/workspace2/moto_imagination/SHU/processed/mat_clean/` (+ npz + processed_manifest.csv) | SHU 全量预处理 125 sessions ok（仓库外，与 WBCIC 同惯例）。 |
| 2026-06-11 20:12:00 | CREATE | `code/configs/experiments/shu_phase{0,1,2b,2c}_*.yaml` | 4 个 SHU 实验 config（32ch/5 session，数据集作用域输出）。 |
| 2026-06-11 20:12:00 | UPDATE | `code/runners.py`, `code/configs/datasets/shu.yaml` | manifest 按 config 解耦（`_resolve_manifest`）；shu.yaml 增 manifest/status_filter。 |
| 2026-06-11 21:20:00 | RENAME | `/share/workspace2/moto_imagination/SHU/processed/mat_clean` -> `.../npz_clean` | 去掉误导性 mat 命名（实际存的是 npz）；同步重写 manifest 路径列与 SHU configs。 |
| 2026-06-11 21:20:00 | MOVE | `checkpoints/prototype_drift_v1` -> `checkpoints/wbci_shu/prototype_drift_v1` | checkpoints 改数据集并列（wbci_shu/ 前缀）。 |
| 2026-06-11 21:20:00 | MOVE | `outputs/experiments/prototype_drift_v1{,_smoke}` -> `outputs/experiments/wbci_shu/` | outputs 改数据集并列；同步更新 5 个 WBCIC config 的 output_dir/checkpoint_dir。 |
| 2026-06-11 21:24:00 | CREATE | `checkpoints/README.md` | checkpoint 命名规范（dataset/run_id/method/model/任务前缀/sub/session/seed），自文档化。 |
| 2026-06-11 21:55:00 | CREATE | `scripts/slurm/shu_gpu.sbatch`, `scripts/slurm/shu_cpu.sbatch` | SHU 通用 Slurm 提交脚本（GPU 训练 / CPU 漂移+汇总，均 mi_torch_cu118，禁登录节点跑 GPU）。 |
| 2026-06-11 21:56:00 | CREATE | `scripts/build_drift_report.py` | session/dataset 无关的 drift per-pair/per-subject 表+图+报告构建器（泛化 legacy WBCIC-only 版）。 |
| 2026-06-11 22:03:00 | CREATE | `outputs/analysis/shu/session_drift_v1/` | SHU Phase 0 漂移诊断全量结果（250 pairs / 25 subjects，Slurm job 21601）。 |
| 2026-06-11 22:05:00 | CREATE | `1_session_drift/shu/{report,tables,figures}/*` + `report/AI_ANALYSIS.md` | SHU Phase 0 可读结果落地 + AI 深度分析（数字源自 tables）。 |
| 2026-06-11 22:05:00 | CREATE | `code/configs/experiments/shu_phase2a_multisource.yaml` | SHU Phase 2a 可比协议 config（ses-01+02→ses-03，复用 runner，未改 runner）。 |
| 2026-06-11 22:06:00 | UPDATE | `code/summaries/summarize.py` | phase2a baseline_summaries 路径改为 config 驱动（数据集作用域，非 runner）。 |
| 2026-06-11 22:07:00 | SUBMIT | `outputs/experiments/shu/{session_model_compare_v1,session_multisource_v1,alignment_baseline_v1,prototype_drift_v1}/` | SHU Phase 1/2a/2b/2c 训练全量提交（Slurm jobs 21602-21604 + 21610-21642）；id 记于 `outputs/experiments/shu/_job_ids/shu_full_2228.txt`。 |
| 2026-06-11 22:44:00 | UPDATE | `code/configs/experiments/shu_phase1_baseline.yaml` | `train.seeds` 由 [0] 改为 [0,1,2,3,4]，对齐 WBCIC 5-seed 标准。 |
| 2026-06-11 22:44:00 | SUBMIT | `outputs/experiments/shu/session_model_compare_v1/` | 补提交 Phase 1 seeds 1-4（Slurm jobs 21644-21655，每 model×seed），per-seed CSV 不覆盖 seed0；id 记于 `outputs/experiments/shu/_job_ids/shu_phase1_seeds1-4_2244.txt`。 |
| 2026-06-11 23:15:00 | APPEND | `outputs/experiments/shu/_job_ids/shu_full_2228.txt` | 末尾追加 Phase 1 seed1-4 补交 job 引用（21644-21655），未改历史行。 |
| 2026-06-12 16:32:00 | CREATE | `HANDOFF.md` | 跨 Agent / 非 Cursor 通用交接文档：项目逻辑、代码逻辑、工具兼容、运行方式、文档更新规范、接手自检清单。 |
| 2026-06-12 16:32:00 | CREATE | `CLAUDE.md` | 本目录兼容入口，指向 AGENTS.md（Claude Code 等会原生读取 CLAUDE.md）。 |
| 2026-07-05 23:40:00 | CREATE | `CHATGPT_DIRECTOR_BRIEF.md` | ChatGPT「指挥官」简报：粘贴进 ChatGPT 让其扮演 PI 指挥 Cursor 执行者；含 2026-07-05 盘面（SHU 训练 done / summarize pending）。 |
| 2026-07-06 00:10:00 | CREATE | `scripts/make_baseline_cross_all.py` | Phase 2b baseline schema 适配器（Phase1 cross → alignment 口径 results_cross_session_all.csv，数据集无关）。 |
| 2026-07-06 00:15:00 | CREATE | `outputs/experiments/shu/session_model_compare_v1/{summaries,cross_session/tables}/*` | SHU P1 summarize 产物（tables/figures/report + baseline_cross_all）。 |
| 2026-07-06 00:15:00 | CREATE | `outputs/experiments/shu/session_multisource_v1/summaries/*` | SHU P2a multisource summarize 产物。 |
| 2026-07-06 00:15:00 | CREATE | `outputs/experiments/shu/alignment_baseline_v1/cross_session/{tables,figures}/*` + reports | SHU P2b alignment summarize 产物。 |
| 2026-07-06 00:16:00 | CREATE | `4_experiments/shu/prototype_drift/{tables,figures,report}/*` | SHU P2c prototype drift summarize 产物（直接写入可读区）。 |
| 2026-07-06 00:20:00 | COPY | `outputs/experiments/shu/{session_model_compare_v1,session_multisource_v1}/summaries` → `2_baseline/shu/no_alignment_baseline/{tables,figures,report}` | SHU P1+P2a 可读结果落地（并入同一目录）。 |
| 2026-07-06 00:20:00 | COPY | `outputs/experiments/shu/alignment_baseline_v1/cross_session` → `2_baseline/shu/alignment_baseline/{tables,figures,report}` | SHU P2b 可读结果落地。 |
| 2026-07-06 00:25:00 | CREATE | `2_baseline/shu/no_alignment_baseline/report/AI_ANALYSIS.md` | SHU P1+P2a AI 深度分析（数字源自 tables）。 |
| 2026-07-06 00:25:00 | CREATE | `2_baseline/shu/alignment_baseline/report/AI_ANALYSIS.md` | SHU P2b AI 深度分析。 |
| 2026-07-06 00:25:00 | CREATE | `4_experiments/shu/prototype_drift/report/AI_ANALYSIS.md` | SHU P2c AI 深度分析。 |
| 2026-07-06 00:32:00 | DELETE | `CHATGPT_DIRECTOR_BRIEF.md` | 用户不需要；核心灵魂集（AGENTS/progress/STATUS/ARCHITECTURE）直接导入 ChatGPT 即可，简报冗余。 |
| 2026-07-06 00:32:00 | NOTE | `HANDOFF.md` | 用户已手动删除（连同上层 P10_MI泛化研究/、根目录 CLAUDE.md、ChatGPT md）；已清理各处对它的引用。 |
| 2026-07-06 01:25:00 | CREATE | `3_online_adaptation/PHASE3_TTA_DESIGN.md` | Phase 3 T3A 测试时适应权威路线文件（方法/矩阵/Step A–F，planned）。 |
| 2026-07-06 01:25:00 | CREATE | `code/configs/experiments/phase3_tta.yaml` | Phase 3 T3A 配置骨架（WBCIC；runner 待实现）。 |
| 2026-07-06 01:25:00 | CREATE | `code/configs/experiments/shu_phase3_tta.yaml` | Phase 3 T3A 配置骨架（SHU）。 |
| 2026-07-06 01:25:00 | UPDATE | `AGENTS.md` / `progress.md` / `0_docs/STATUS.md` / `0_docs/FILE_CATALOG.md` | Phase 3 T3A 立项：更新主线/逻辑链/事实/进度表/下一步/文件索引（参考 arXiv:2604.16926v1）。 |
| 2026-07-08 23:20:00 | CREATE | `PHASE3_ROUTE_PLAN.md` | Phase 3 完整路线计划 v2（Oracle 先裁决，待审批）。 |
| 2026-07-08 23:20:00 | UPDATE | `3_online_adaptation/PHASE3_TTA_DESIGN.md` | 重排实现路线：Oracle 从 Step E 提前为 Phase B 裁决门。 |
| 2026-07-08 23:20:00 | UPDATE | `AGENTS.md` / `progress.md` / `0_docs/STATUS.md` / `README.md` / `results.md` | Phase 0 状态修正：路线 v2（Oracle 先裁决），标注 embed_index 失效路径 bug。 |
| 2026-07-08 23:45:00 | CREATE | `PROJECT_ARCH_SYNC_FOR_ADVISOR.md` | 给学长的项目架构同步简报（新旧结构映射，防冲突）。 |
| 2026-07-08 23:55:00 | UPDATE | `PHASE3_ROUTE_PLAN.md` | 升级 v2.1 批准版：学长批准 A0/A1 + 新增 §2.5 七条硬约束并逐 Phase 落实。 |
| 2026-07-08 23:55:00 | UPDATE | `code/configs/experiments/{phase3_tta,shu_phase3_tta}.yaml` | 硬约束1：readable_dir 由 3_online_adaptation/*/tta 改为 4_experiments/*/tta。 |
| 2026-07-08 23:55:00 | UPDATE | `3_online_adaptation/PHASE3_TTA_DESIGN.md` / `AGENTS.md` / `progress.md` / `0_docs/STATUS.md` | 同步批准状态 + 7 条硬约束 + 结果目录定死。 |
