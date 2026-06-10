---
title: "Completed Artifacts Index"
tags:
  - "#pipeline/4_analysis"
  - "#modality/eeg"
created: "2026-06-10"
updated: "2026-06-10"
status: "active"
---

# Completed Artifacts Index

This file records where completed results live after the root cleanup. The original root-level
`outputs/`, `checkpoints/`, and `logs/` folders were moved into
`backup/root_archive_2026-06-10/` so the project root matches the P10-style layout.

| Path | Approx Size | Role | Move Policy |
|:---|:---|:---|:---|
| `backup/root_archive_2026-06-10/outputs/analysis/session_drift_v1/` | part of outputs | Phase 0 drift diagnostic outputs | Archived; do not delete. |
| `backup/root_archive_2026-06-10/outputs/experiments/baseline_v1/` | part of outputs | Canonical Phase 1 baseline report/tables/figures | Archived; do not delete. |
| `backup/root_archive_2026-06-10/outputs/experiments/session_model_compare_v1/` | part of outputs | Raw/provenance baseline run outputs | Archived; do not delete. |
| `backup/root_archive_2026-06-10/outputs/experiments/session_multisource_v1/` | part of outputs | Phase 2a multi-source outputs | Archived; do not delete. |
| `backup/root_archive_2026-06-10/outputs/experiments/alignment_baseline_v1/` | part of outputs | Phase 2b alignment outputs | Archived; do not delete. |
| `backup/root_archive_2026-06-10/checkpoints/session_model_compare_v1/` | part of checkpoints | Baseline weights | Archived; large. |
| `backup/root_archive_2026-06-10/checkpoints/session_multisource_v1/` | part of checkpoints | Multi-source weights | Archived; large. |
| `backup/root_archive_2026-06-10/checkpoints/alignment_baseline_v1/` | part of checkpoints | Alignment weights | Archived; large. |
| `backup/root_archive_2026-06-10/logs/slurm/` | part of logs | Slurm stdout/stderr history | Archived. |

If a future experiment needs these as inputs, point explicitly to the archived paths above or
restore a compatibility layer before running.
