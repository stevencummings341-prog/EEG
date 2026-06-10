# Session Model Comparison Report — session_model_compare_v1

Baselines: **EEGNet / DeepConvNet / FBCNet** under one protocol, data filter (status=ok, 148 sessions), and metric set. Seeds aggregated as mean±std.

## 1. Within-session 10-fold CV (mean ± std across seeds)

| model | n_seeds | Acc | BalAcc | MacroF1 | AUC | NLL | Brier | ECE | median Acc | min | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `deepconvnet` | 5 | 0.766±0.002 | 0.766±0.002 | 0.759±0.002 | 0.825±0.003 | 0.521±0.006 | 0.315±0.003 | 0.171±0.002 | 0.767 | 0.763 | 0.768 |
| `eegnet` | 5 | 0.807±0.002 | 0.807±0.002 | 0.803±0.002 | 0.859±0.001 | 0.426±0.002 | 0.259±0.001 | 0.142±0.001 | 0.808 | 0.803 | 0.809 |
| `fbcnet` | 5 | 0.720±0.003 | 0.720±0.003 | 0.714±0.003 | 0.772±0.003 | 0.535±0.003 | 0.358±0.003 | 0.190±0.002 | 0.722 | 0.715 | 0.723 |

### Within-session by session (Acc mean ± std across seeds)

| model | ses-01 | ses-02 | ses-03 |
|---|---|---|---|
| `deepconvnet` | 0.744±0.004 | 0.767±0.002 | 0.787±0.002 |
| `eegnet` | 0.779±0.004 | 0.817±0.001 | 0.823±0.004 |
| `fbcnet` | 0.709±0.004 | 0.716±0.004 | 0.736±0.003 |

## 2. Cross-session (mean ± std across seeds)

| model | n_seeds | Acc | BalAcc | MacroF1 | AUC | NLL | Brier | ECE | median Acc | min | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `deepconvnet` | 5 | 0.681±0.002 | 0.681±0.002 | 0.666±0.004 | 0.741±0.003 | 0.874±0.016 | 0.447±0.003 | 0.160±0.003 | 0.681 | 0.677 | 0.684 |
| `eegnet` | 5 | 0.711±0.008 | 0.711±0.008 | 0.705±0.008 | 0.761±0.008 | 0.822±0.023 | 0.405±0.011 | 0.137±0.004 | 0.708 | 0.702 | 0.722 |
| `fbcnet` | 5 | 0.628±0.003 | 0.628±0.003 | 0.594±0.004 | 0.690±0.004 | 0.886±0.017 | 0.514±0.004 | 0.186±0.002 | 0.628 | 0.623 | 0.631 |

### Cross-session by direction (Acc mean ± std across seeds)

| model | train → test | Acc |
|---|---|---|
| `deepconvnet` | ses-01 → ses-02 | 0.679±0.005 |
| `deepconvnet` | ses-01 → ses-03 | 0.676±0.004 |
| `deepconvnet` | ses-02 → ses-01 | 0.678±0.008 |
| `deepconvnet` | ses-02 → ses-03 | 0.721±0.009 |
| `deepconvnet` | ses-03 → ses-01 | 0.652±0.005 |
| `deepconvnet` | ses-03 → ses-02 | 0.680±0.006 |
| `eegnet` | ses-01 → ses-02 | 0.714±0.013 |
| `eegnet` | ses-01 → ses-03 | 0.699±0.009 |
| `eegnet` | ses-02 → ses-01 | 0.707±0.008 |
| `eegnet` | ses-02 → ses-03 | 0.749±0.008 |
| `eegnet` | ses-03 → ses-01 | 0.681±0.008 |
| `eegnet` | ses-03 → ses-02 | 0.712±0.005 |
| `fbcnet` | ses-01 → ses-02 | 0.617±0.004 |
| `fbcnet` | ses-01 → ses-03 | 0.614±0.006 |
| `fbcnet` | ses-02 → ses-01 | 0.632±0.003 |
| `fbcnet` | ses-02 → ses-03 | 0.648±0.005 |
| `fbcnet` | ses-03 → ses-01 | 0.617±0.005 |
| `fbcnet` | ses-03 → ses-02 | 0.639±0.005 |

## 3. Within vs Cross-session

| model | within Acc | cross Acc | drop | relative drop |
|---|---|---|---|---|
| `eegnet` | 0.8067 | 0.7105 | 0.0961 | 11.9% |
| `deepconvnet` | 0.7663 | 0.6811 | 0.0852 | 11.1% |
| `fbcnet` | 0.7203 | 0.6280 | 0.0923 | 12.8% |

## 4. Paper baseline comparison (within-session 10-fold CV)

WBCIC-SHU paper reports (within-session 10-fold CV accuracy %): EEGNet 85.32, DeepConvNet 84.47, FBCNet 78.40.

| model | ours within Acc (%) | paper (%) | Δ (ours − paper, pp) |
|---|---|---|---|
| `eegnet` | 80.67 | 85.32 | -4.65 |
| `deepconvnet` | 76.63 | 84.47 | -7.84 |
| `fbcnet` | 72.03 | 78.40 | -6.37 |

**Session trend (EEGNet, ours vs paper, Acc %):** paper S1 81.77 / S2 86.63 / S3 88.90.

| session | ours (%) | paper (%) |
|---|---|---|
| ses-01 | 77.94 | 81.77 |
| ses-02 | 81.74 | 86.63 |
| ses-03 | 82.28 | 88.90 |

## 5. Reliability checks

- Completed (model, protocol, seed) cells: **30** (target 30 = 3 models × 2 protocols × 5 seeds).
- Within: distinct (subject,session) used = **148** (all 148 used: True); expected rows per (model,seed) = 1480.
- Cross: distinct directed pairs = **288** (expected 288); invalid same-session pairs = 0 (must be 0 — no leakage).
- NaN metric cells: none.
- Leakage: by construction within=disjoint folds (val carved from train only), cross=different sessions; no test trials in train/val.

## 6. Figures

- `within_session_accuracy_boxplot.png`
- `cross_session_accuracy_matrix_by_model.png`
- `protocol_comparison.png`

## Notes

- Mainline = the THREE baseline architectures. CAP-EEGNet (v1/v2) and all agent/toolkit/prototype/confidence/online/fine-tuning modules remain FUTURE work.
- LOSO and 41/10 are not run.
