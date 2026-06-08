# Experiment Log (project)

Newest entries on top. Auto + manual entries. Full narrative + decisions live in
`docs/PROGRESS.md`; consolidated results in `docs/RESULTS_SUMMARY.md`.

## 2026-06-08 — Step 1 multi-source static baseline COMPLETE (`ses-01+02 → ses-03`)

- Protocol: same-subject `train = ses-01 + ses-02`, `test = ses-03`; val carved only from the
  combined train; ses-03 labels used only for final eval.
- Models: EEGNet / DeepConvNet / FBCNet. Seeds: 0,1,2,3,4. Data: `eog_ecg_clean` status=ok.
- Eligible subjects: 47 (all 3 sessions ok); skipped: `sub-023`, `sub-024`, `sub-026`, `sub-032`.
- Smoke test: subjects 1,2, EEGNet, seed 0, 3 epochs, CUDA — passed (n_train=320/n_val=80/n_test=200,
  no leakage, no NaN).
- Full run jobs: `21240,21241,21242,21243,21244` (train, one per seed) + `21245` (summarize);
  all COMPLETED exit `0:0`. 705 rows ok / 0 failed / 0 NaN.

| model | ses-01→03 | ses-02→03 | ses-01+02→03 | Δ vs best single |
|---|---:|---:|---:|---:|
| EEGNet | 0.6991±0.009 | 0.7492±0.008 | **0.7717±0.003** | +0.0224 |
| DeepConvNet | 0.6757±0.004 | 0.7211±0.009 | **0.7564±0.007** | +0.0353 |
| FBCNet | 0.6142±0.006 | 0.6484±0.005 | **0.6750±0.002** | +0.0267 |

- Multi-source beats the best single source for all 3 models (mean Δ +0.0281); beats even a
  per-subject oracle single source; recovers ~30–53% of the cross→within-ses-03 gap.
- Full analysis: `docs/MULTISOURCE_STEP1_REPORT.md`; outputs in
  `outputs/experiments/baseline_v1/provenance/session_multisource_v1/summaries/`.
- Step 2 no-learning adaptation remains next and was **NOT run**. online / 41-10 / fine-tuning /
  CAP-EEGNet full / multi-agent / prototype / memory remain future work.

## 2026-06-08 — P10 integration / docs only (NO experiment run)

Integrated the senior's P10 package into the project's docs/direction (Phase 0/1 match this
project; Phase 2 online framework = draft, NOT validated → future). No training/sbatch/GPU run,
no raw/workspace2 writes.

## 2026-06-07 — Baseline 5-seed run COMPLETE (within CV + single-source cross)

- 30/30 cells (3 models × 2 protocols × 5 seeds), 26 520 rows, no NaN, no leakage.
- within Acc: EEGNet 0.807±0.002 / DeepConvNet 0.766±0.002 / FBCNet 0.720±0.003.
- single-source cross Acc: EEGNet 0.711±0.008 / DeepConvNet 0.681±0.002 / FBCNet 0.628±0.003.
  Cross-session drop 11.9 / 11.1 / 12.8%.
- vs paper within: −4.65 / −7.84 / −6.37 pp (training-recipe/data-budget gap, not architecture bug).
- Jobs 21161–21171. Full analysis: `docs/RESULTS_SUMMARY.md`; report in
  `outputs/experiments/baseline_v1/provenance/session_model_compare_v1/summaries/`.

## 2026-06-06 — Session drift diagnostic COMPLETE

- 144 within-subject session pairs / 50 subjects. MMD 0.238, CSP 0.420, ERD-μ 0.419, μ-KS 0.246,
  RMS median 0.992, Fisher≈0 → drift is spatial + μ/β spectral, not amplitude.
- Outputs `outputs/analysis/session_drift_v1/`; see `docs/SESSION_DRIFT_ANALYSIS.md`.

## Prior (data) milestones

- Full `eog_ecg_clean` preprocessing of 51×3: **148 ok / 5 failed**; QC vs official derivatives PASS.
- Preprocessing pipeline + dataset + manifests + 41/10 splits (41/10 = future) in place.
