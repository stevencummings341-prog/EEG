# Multi-source Cross-session Baseline — Step 1 Report (multisource_0102_to_03)

Protocol: **train = ses-01+ses-02** (same subject, concatenated trials) -> **test = ses-03** (all trials). Models EEGNet / DeepConvNet / FBCNet, seeds 0-4, data = `eog_ecg_clean` status=ok only.

## 1. Protocol & no-leakage design

- train set = ALL trials of ses-01+ses-02 for the subject (combined).
- val set = stratified slice carved **only from train**; never contains ses-03 trials.
- test set = ALL trials of ses-03; used ONLY for final evaluation.
- per-row `n_train` / `n_val` / `n_test` and checkpoint path are recorded.

## 2. Used vs skipped subjects

- **Used subjects (ses-01/02/03 all ok): 47** — sub-001, sub-002, sub-003, sub-004, sub-005, sub-006, sub-007, sub-008, sub-009, sub-010, sub-011, sub-012, sub-013, sub-014, sub-015, sub-016, sub-017, sub-018, sub-019, sub-020, sub-021, sub-022, sub-025, sub-027, sub-028, sub-029, sub-030, sub-031, sub-033, sub-034, sub-035, sub-036, sub-037, sub-038, sub-039, sub-040, sub-041, sub-042, sub-043, sub-044, sub-045, sub-046, sub-047, sub-048, sub-049, sub-050, sub-051
- **Skipped subjects: 4**

| subject | reason | ok_sessions |
|---|---|---|
| sub-023 | missing ok session(s): ses-01 | ses-02|ses-03 |
| sub-024 | missing ok session(s): ses-02,ses-03 | ses-01 |
| sub-026 | missing ok session(s): ses-01 | ses-02|ses-03 |
| sub-032 | missing ok session(s): ses-02 | ses-01|ses-03 |

## 3. Results — mean ± std across seeds (test on ses-03)

| model | Acc | BalAcc | MacroF1 | AUC | NLL | Brier | ECE | n_seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `eegnet` | 0.7717±0.003 | 0.7717 | 0.7663 | 0.8258 | 0.5522 | 0.3161 | 0.1020 | 5 |
| `deepconvnet` | 0.7564±0.007 | 0.7564 | 0.7472 | 0.8175 | 0.6160 | 0.3429 | 0.1179 | 5 |
| `fbcnet` | 0.6750±0.002 | 0.6750 | 0.6488 | 0.7392 | 0.7398 | 0.4401 | 0.1484 | 5 |

## 4. Comparison vs single-source cross-session (test = ses-03)

| model | ses-01->03 | ses-02->03 | **ses-01+02->03** | Δ vs best single |
|---|---:|---:|---:|---:|
| `eegnet` | 0.6991±0.009 | 0.7492±0.008 | **0.7717±0.003** | +0.0224 |
| `deepconvnet` | 0.6757±0.004 | 0.7211±0.009 | **0.7564±0.007** | +0.0353 |
| `fbcnet` | 0.6142±0.006 | 0.6484±0.005 | **0.6750±0.002** | +0.0267 |

- Mean Δ over models vs best single source = **+0.0281** -> **multi-source HELPS**.

## 5. Worse-than-single-source check

No (model, seed) fell below the strongest single-source `ses-02->03` mean.

## 6. No-leakage / reliability checks

- Result rows: 705 (ok=705, failed=0).
- NaN accuracy among ok rows: 0.
- Code guards: test session not in train sessions; train/val disjoint; val carved only from ses-01+02.
- n_train range: [320, 320]; n_val range: [80, 80]; n_test range: [200, 200].

## 7. Next step (NOT run here)

- Step 2 = no-learning adaptation baseline: none / session_zscore / Euclidean Alignment / Riemannian Alignment / target BN-stats / filter-bank reweighting.
- online / 41-10 / fine-tuning / CAP-EEGNet full / multi-agent / prototype / memory remain future work.

## 8. Files

- raw results: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/session_multisource_v1/summaries/results_multisource_0102_to_03.csv`
- by seed: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/session_multisource_v1/summaries/multisource_by_seed.csv`
- by model: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/session_multisource_v1/summaries/multisource_by_model.csv`
- by model/protocol: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/session_multisource_v1/summaries/summary_by_model_protocol.csv`
- report: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/session_multisource_v1/summaries/MULTISOURCE_STEP1_REPORT.md`
- figure: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/session_multisource_v1/figures/multisource_vs_singlesource_acc.png`
