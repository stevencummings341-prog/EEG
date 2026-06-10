# BASELINE_REPORT.md

## 1. Executive Summary

- within-session CV is the no-drift upper bound.
- single-source cross-session shows a 9-13% relative drop.
- multi-source `ses-01+02 -> ses-03` improves over the best single source for all three models.
- Some subjects still degrade under naive multi-source, motivating Step 2 no-learning alignment.

## 2. Dataset and Protocol

- Data: `eog_ecg_clean`, status=ok only (148 ok / 5 failed).
- Models: EEGNet / DeepConvNet / FBCNet.
- Seeds: 0-4.
- Multi-source uses 47 subjects with all three sessions ok.
- No leakage: validation is carved only from train; test labels are used only for final evaluation.

Canonical tables:
- within: `within_session/tables/`
- cross: `cross_session/tables/results_cross_session_all.csv`
- cross protocol comparison: `cross_session/tables/cross_protocol_comparison.csv`
- per-subject gains: `cross_session/tables/cross_by_subject.csv`

## 3. Within-session Baseline

| model | Acc |
|---|---:|
| EEGNet | 0.807±0.002 |
| DeepConvNet | 0.766±0.002 |
| FBCNet | 0.720±0.003 |

![within trend](within_session/figures/within_session_trend_by_model.png)

## 4. Single-source Cross-session Baseline

| model | Cross Acc | Drop vs within |
|---|---:|---:|
| EEGNet | 0.711±0.008 | 0.096 |
| DeepConvNet | 0.681±0.002 | 0.085 |
| FBCNet | 0.628±0.003 | 0.092 |

![direction matrix](cross_session/figures/cross_session_accuracy_matrix_by_model.png)

## 5. Multi-source Cross-session Baseline

| model | ses-01->03 | ses-02->03 | ses-01+02->03 | gain vs best single |
|---|---:|---:|---:|---:|
| EEGNet | 0.6991±0.009 | 0.7492±0.008 | **0.7717±0.003** | +0.0224 |
| DeepConvNet | 0.6757±0.004 | 0.7211±0.009 | **0.7564±0.007** | +0.0353 |
| FBCNet | 0.6142±0.006 | 0.6484±0.005 | **0.6750±0.002** | +0.0267 |

![single vs multisource](cross_session/figures/single_vs_multisource_accuracy.png)

## 6. Gap Recovery Analysis

See `cross_session/tables/cross_gap_recovery.csv`.

![gap recovery](cross_session/figures/gap_recovery_by_model.png)

## 7. Per-subject Gain

See `cross_session/tables/cross_by_subject.csv`.
This table has 150 rows (3 models × 50 subjects with comparable single-source `test=ses-03`);
multi-source fields are filled for the 47 subjects whose three sessions are all ok.

![gain by subject](cross_session/figures/multisource_gain_by_subject.png)

## 8. Reliability and Leakage Checks

- within rows: 22,200.
- single-source cross rows: 4,320.
- multi-source rows: 705 (0 failed, 0 NaN).
- multi-source sizes: n_train=320, n_val=80, n_test=200.
- Splits saved under `within_session/splits/` and `cross_session/splits/`.

## 9. Next Stage

Step 2 no-learning adaptation baseline: none / session_zscore / EA / Riemannian / BN stats / filter-bank.
