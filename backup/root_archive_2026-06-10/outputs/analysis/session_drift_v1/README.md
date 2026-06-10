# session_drift_v1

Session-drift diagnosis results computed from the preprocessed `eog_ecg_clean` `.npz` files.
This folder contains **data-level drift analysis only**; it does not contain model training
or evaluation results.

## Contents

- `SESSION_DRIFT_REPORT.md` — full drift report.
- `SESSION_DRIFT_SUMMARY_CN.md` — Chinese one-page summary.
- `session_drift_report.csv` — one row per within-subject session pair (144 pairs).
- `session_pair_summary.csv` — aggregate by session pair (01-02, 01-03, 02-03).
- `per_subject_drift_summary.csv` / `.md` — subject-level drift profile and drift_level.
- `summary.json` — global metric summary.
- `figures/` — drift figures (MMD, CSP, ERD/ERS, band power, RMS, Fisher, subject heatmaps).

## Key conclusion

Cross-session drift is mainly spatial + mu/beta spectral:

- MMD ≈ 0.238
- CSP similarity ≈ 0.420
- ERD-mu corr ≈ 0.419
- mu-KS ≈ 0.246
- RMS median ≈ 0.992 (amplitude is not the main cause)

These findings motivate Step 2 no-learning adaptation (EA / Riemannian / BN stats / filter-bank).
