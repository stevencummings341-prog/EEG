# Alignment Baseline (Step 2) — RUN PLAN

## Scope
No-learning / unsupervised test-time alignment baseline. Source-trained model;
target used only via unlabeled X; `y_test` only for final eval; no `optimizer.step`
on target (BN running-stat update only for the BN method).

## Grid
- **Methods (trained):** `session_zscore`, `euclidean_alignment`, `riemannian_alignment`,
  `bn_statistics_adaptation`, `filterbank_reweighting` (5).
  `none_reference` is NOT trained — pulled from baseline_v1 by the summarizer.
- **Models:** eegnet, deepconvnet, fbcnet (3).
- **Seeds:** 0,1,2,3,4 (5).
- **Protocols:**
  - single-source directed pairs: **288** pairs (per model per seed) — all directions where
    train & test session both `status=ok`.
  - multi-source ses-01+ses-02 → ses-03: **47** subjects (4 skipped: missing one of ses-01/02/03).

## Estimated training count
Per (method, model, seed): 288 single + 47 multi = **335** trainings.
Total: 5 methods × 3 models × 5 seeds × 335 = **25,125 trainings**.

## Estimated GPU time
- Reference: baseline single-source cross (~864 trainings/seed-job over 3 models) completed
  within the 12 h wall on `gpu2node`.
- Each alignment job here = one (method, model, seed) = 335 trainings + alignment overhead
  (per-trial covariance / FIR band decomposition, CPU): EEGNet ≈ 1 h, FBCNet ≈ 2 h,
  DeepConvNet ≈ 3–5 h. All comfortably < 12 h.

## Job layout
- **75 GPU jobs** = method × model × seed (each covers BOTH protocol groups, single+multi).
  Granularity keeps every job small per the run guidance.
- Partition `gpu2node`, `--gres=gpu:1`, env `mi_torch_cu118`, fail-fast if CUDA unavailable.
- Each job writes collision-free CSVs `cross_session/runs/alignment__{method}__{model}__{scope}__seed{seed}__{suffix}.csv`
  and a unique `meta_alignment__{suffix}.json`. Split JSONs (`cross_session/splits/`) are
  written atomically (shared per task+seed, last-writer-wins, identical content).
- Logs → `logs/slurm/`. Job ids → `full_job_ids.txt`.

## Dependent summarizer
- 1 CPU job, `--dependency=afterany:<all 75 ids>` → `scripts/summarize_alignment_results.py`.
- Produces `cross_session/tables/*`, `cross_session/figures/*`,
  `ALIGNMENT_BASELINE_REPORT.md`, `RUN_STATUS.md`, `manifest_sources.json`.
- Pulls `none_reference` from
  `outputs/experiments/baseline_v1/cross_session/tables/results_cross_session_all.csv`
  and drift levels from `outputs/analysis/session_drift_v1/per_subject_drift_summary.csv`.

## Do-NOT
No online, no 41/10, no LOSO, no fine-tuning, no CAP-EEGNet full, no multi-agent/prototype/
memory, no new deps, no edits to raw/workspace2, no overwrite of baseline_v1.
