---
title: "Operation Log"
tags:
  - "#modality/eeg"
created: "2026-06-10"
updated: "2026-06-10"
status: "active"
---

# Operation Log

| Time | Action | Path | Note |
|:---|:---|:---|:---|
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
