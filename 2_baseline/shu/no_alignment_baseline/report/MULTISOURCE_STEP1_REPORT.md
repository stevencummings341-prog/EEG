# Multi-source Cross-session Baseline — Step 1 Report (multisource_0102_to_03)

Protocol: **train = ses-01+ses-02** (same subject, concatenated trials) -> **test = ses-03** (all trials). Models EEGNet / DeepConvNet / FBCNet, seeds 0-4, data = `eog_ecg_clean` status=ok only.

## 1. Protocol & no-leakage design

- train set = ALL trials of ses-01+ses-02 for the subject (combined).
- val set = stratified slice carved **only from train**; never contains ses-03 trials.
- test set = ALL trials of ses-03; used ONLY for final evaluation.
- per-row `n_train` / `n_val` / `n_test` and checkpoint path are recorded.

## 2. Used vs skipped subjects

- **Used subjects (ses-01/02/03 all ok): 25** — sub-001, sub-002, sub-003, sub-004, sub-005, sub-006, sub-007, sub-008, sub-009, sub-010, sub-011, sub-012, sub-013, sub-014, sub-015, sub-016, sub-017, sub-018, sub-019, sub-020, sub-021, sub-022, sub-023, sub-024, sub-025
- **Skipped subjects: 0**

## 3. Results — mean ± std across seeds (test on ses-03)

| model | Acc | BalAcc | MacroF1 | AUC | NLL | Brier | ECE | n_seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `eegnet` | 0.5436±0.013 | 0.5444 | 0.5232 | 0.5656 | 1.0181 | 0.5552 | 0.1592 | 5 |
| `deepconvnet` | 0.5578±0.012 | 0.5588 | 0.5356 | 0.5826 | 1.1181 | 0.5735 | 0.1980 | 5 |
| `fbcnet` | 0.5116±0.007 | 0.5120 | 0.4509 | 0.5273 | 1.0279 | 0.6402 | 0.2425 | 5 |

## 4. Comparison vs single-source cross-session (test = ses-03)

| model | ses-01->03 | ses-02->03 | **ses-01+02->03** | Δ vs best single |
|---|---:|---:|---:|---:|
| `eegnet` | 0.5375±0.011 | 0.5322±0.007 | **0.5436±0.013** | +0.0061 |
| `deepconvnet` | 0.5323±0.007 | 0.5306±0.009 | **0.5578±0.012** | +0.0255 |
| `fbcnet` | 0.5143±0.007 | 0.5029±0.005 | **0.5116±0.007** | -0.0027 |

- Mean Δ over models vs best single source = **+0.0096** -> **multi-source HELPS**.

## 5. Worse-than-single-source check

No (model, seed) fell below the strongest single-source `ses-02->03` mean.

## 6. No-leakage / reliability checks

- Result rows: 375 (ok=375, failed=0).
- NaN accuracy among ok rows: 0.
- Code guards: test session not in train sessions; train/val disjoint; val carved only from ses-01+02.
- n_train range: [124, 160]; n_val range: [32, 40]; n_test range: [83, 100].

## 7. Next step (NOT run here)

- Step 2 = no-learning adaptation baseline: none / session_zscore / Euclidean Alignment / Riemannian Alignment / target BN-stats / filter-bank reweighting.
- online / 41-10 / fine-tuning / CAP-EEGNet full / multi-agent / prototype / memory remain future work.

## 8. Files

- raw results: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/shu/session_multisource_v1/summaries/results_multisource_0102_to_03.csv`
- by seed: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/shu/session_multisource_v1/summaries/multisource_by_seed.csv`
- by model: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/shu/session_multisource_v1/summaries/multisource_by_model.csv`
- by model/protocol: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/shu/session_multisource_v1/summaries/summary_by_model_protocol.csv`
- report: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/shu/session_multisource_v1/summaries/MULTISOURCE_STEP1_REPORT.md`
- figure: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/shu/session_multisource_v1/figures/multisource_vs_singlesource_acc.png`
