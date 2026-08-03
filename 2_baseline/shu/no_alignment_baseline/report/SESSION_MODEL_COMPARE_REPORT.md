# Session Model Comparison Report — session_model_compare_v1

> ⚠️ **INCOMPLETE**: not all expected runs are present (see Reliability). Numbers below are computed on the available data only.

Baselines: **EEGNet / DeepConvNet / FBCNet** under one protocol, data filter (status=ok, 148 sessions), and metric set. Seeds aggregated as mean±std.

## 1. Within-session 10-fold CV (mean ± std across seeds)

| model | n_seeds | Acc | BalAcc | MacroF1 | AUC | NLL | Brier | ECE | median Acc | min | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `deepconvnet` | 5 | 0.606±0.004 | 0.607±0.003 | 0.570±0.003 | 0.637±0.004 | 0.718±0.010 | 0.469±0.005 | 0.255±0.004 | 0.608 | 0.602 | 0.610 |
| `eegnet` | 5 | 0.611±0.004 | 0.611±0.004 | 0.589±0.004 | 0.638±0.004 | 0.669±0.002 | 0.448±0.002 | 0.223±0.004 | 0.612 | 0.604 | 0.618 |
| `fbcnet` | 5 | 0.553±0.006 | 0.553±0.006 | 0.524±0.006 | 0.572±0.006 | 0.757±0.005 | 0.534±0.005 | 0.317±0.003 | 0.554 | 0.546 | 0.563 |

### Within-session by session (Acc mean ± std across seeds)

| model | ses-01 | ses-02 | ses-03 |
|---|---|---|---|
| `deepconvnet` | 0.606±0.007 | 0.606±0.005 | 0.598±0.008 |
| `eegnet` | 0.610±0.010 | 0.610±0.007 | 0.601±0.011 |
| `fbcnet` | 0.543±0.004 | 0.569±0.007 | 0.531±0.009 |

## 2. Cross-session (mean ± std across seeds)

| model | n_seeds | Acc | BalAcc | MacroF1 | AUC | NLL | Brier | ECE | median Acc | min | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `deepconvnet` | 5 | 0.536±0.003 | 0.536±0.004 | 0.485±0.005 | 0.563±0.005 | 1.080±0.050 | 0.581±0.006 | 0.195±0.009 | 0.539 | 0.531 | 0.539 |
| `eegnet` | 5 | 0.538±0.005 | 0.538±0.005 | 0.516±0.005 | 0.553±0.006 | 1.094±0.053 | 0.561±0.004 | 0.163±0.006 | 0.539 | 0.530 | 0.543 |
| `fbcnet` | 5 | 0.508±0.001 | 0.508±0.001 | 0.420±0.005 | 0.521±0.002 | 1.296±0.040 | 0.691±0.008 | 0.287±0.006 | 0.507 | 0.506 | 0.510 |

### Cross-session by direction (Acc mean ± std across seeds)

| model | train → test | Acc |
|---|---|---|
| `deepconvnet` | ses-01 → ses-02 | 0.532±0.006 |
| `deepconvnet` | ses-01 → ses-03 | 0.532±0.007 |
| `deepconvnet` | ses-01 → ses-04 | 0.535±0.006 |
| `deepconvnet` | ses-01 → ses-05 | 0.541±0.008 |
| `deepconvnet` | ses-02 → ses-01 | 0.536±0.013 |
| `deepconvnet` | ses-02 → ses-03 | 0.531±0.009 |
| `deepconvnet` | ses-02 → ses-04 | 0.548±0.008 |
| `deepconvnet` | ses-02 → ses-05 | 0.543±0.006 |
| `deepconvnet` | ses-03 → ses-01 | 0.520±0.004 |
| `deepconvnet` | ses-03 → ses-02 | 0.515±0.011 |
| `deepconvnet` | ses-03 → ses-04 | 0.511±0.015 |
| `deepconvnet` | ses-03 → ses-05 | 0.531±0.011 |
| `deepconvnet` | ses-04 → ses-01 | 0.531±0.012 |
| `deepconvnet` | ses-04 → ses-02 | 0.559±0.006 |
| `deepconvnet` | ses-04 → ses-03 | 0.530±0.005 |
| `deepconvnet` | ses-04 → ses-05 | 0.569±0.007 |
| `deepconvnet` | ses-05 → ses-01 | 0.528±0.007 |
| `deepconvnet` | ses-05 → ses-02 | 0.544±0.013 |
| `deepconvnet` | ses-05 → ses-03 | 0.538±0.013 |
| `deepconvnet` | ses-05 → ses-04 | 0.552±0.011 |
| `eegnet` | ses-01 → ses-02 | 0.526±0.011 |
| `eegnet` | ses-01 → ses-03 | 0.537±0.011 |
| `eegnet` | ses-01 → ses-04 | 0.545±0.010 |
| `eegnet` | ses-01 → ses-05 | 0.552±0.015 |
| `eegnet` | ses-02 → ses-01 | 0.529±0.008 |
| `eegnet` | ses-02 → ses-03 | 0.532±0.007 |
| `eegnet` | ses-02 → ses-04 | 0.549±0.004 |
| `eegnet` | ses-02 → ses-05 | 0.533±0.011 |
| `eegnet` | ses-03 → ses-01 | 0.528±0.019 |
| `eegnet` | ses-03 → ses-02 | 0.513±0.006 |
| `eegnet` | ses-03 → ses-04 | 0.513±0.007 |
| `eegnet` | ses-03 → ses-05 | 0.534±0.005 |
| `eegnet` | ses-04 → ses-01 | 0.544±0.009 |
| `eegnet` | ses-04 → ses-02 | 0.565±0.013 |
| `eegnet` | ses-04 → ses-03 | 0.538±0.016 |
| `eegnet` | ses-04 → ses-05 | 0.564±0.011 |
| `eegnet` | ses-05 → ses-01 | 0.532±0.011 |
| `eegnet` | ses-05 → ses-02 | 0.530±0.006 |
| `eegnet` | ses-05 → ses-03 | 0.540±0.014 |
| `eegnet` | ses-05 → ses-04 | 0.557±0.009 |
| `fbcnet` | ses-01 → ses-02 | 0.506±0.006 |
| `fbcnet` | ses-01 → ses-03 | 0.514±0.007 |
| `fbcnet` | ses-01 → ses-04 | 0.503±0.006 |
| `fbcnet` | ses-01 → ses-05 | 0.501±0.005 |
| `fbcnet` | ses-02 → ses-01 | 0.505±0.006 |
| `fbcnet` | ses-02 → ses-03 | 0.503±0.005 |
| `fbcnet` | ses-02 → ses-04 | 0.518±0.005 |
| `fbcnet` | ses-02 → ses-05 | 0.512±0.006 |
| `fbcnet` | ses-03 → ses-01 | 0.510±0.004 |
| `fbcnet` | ses-03 → ses-02 | 0.504±0.005 |
| `fbcnet` | ses-03 → ses-04 | 0.498±0.006 |
| `fbcnet` | ses-03 → ses-05 | 0.506±0.004 |
| `fbcnet` | ses-04 → ses-01 | 0.510±0.002 |
| `fbcnet` | ses-04 → ses-02 | 0.520±0.007 |
| `fbcnet` | ses-04 → ses-03 | 0.502±0.006 |
| `fbcnet` | ses-04 → ses-05 | 0.518±0.006 |
| `fbcnet` | ses-05 → ses-01 | 0.503±0.002 |
| `fbcnet` | ses-05 → ses-02 | 0.511±0.005 |
| `fbcnet` | ses-05 → ses-03 | 0.503±0.006 |
| `fbcnet` | ses-05 → ses-04 | 0.508±0.005 |

## 3. Within vs Cross-session

| model | within Acc | cross Acc | drop | relative drop |
|---|---|---|---|---|
| `eegnet` | 0.6113 | 0.5381 | 0.0732 | 12.0% |
| `deepconvnet` | 0.6064 | 0.5363 | 0.0701 | 11.6% |
| `fbcnet` | 0.5535 | 0.5078 | 0.0457 | 8.2% |

## 4. Paper baseline comparison (within-session 10-fold CV)

WBCIC-SHU paper reports (within-session 10-fold CV accuracy %): EEGNet 85.32, DeepConvNet 84.47, FBCNet 78.40.

| model | ours within Acc (%) | paper (%) | Δ (ours − paper, pp) |
|---|---|---|---|
| `eegnet` | 61.13 | 85.32 | -24.19 |
| `deepconvnet` | 60.64 | 84.47 | -23.83 |
| `fbcnet` | 55.35 | 78.40 | -23.05 |

**Session trend (EEGNet, ours vs paper, Acc %):** paper S1 81.77 / S2 86.63 / S3 88.90.

| session | ours (%) | paper (%) |
|---|---|---|
| ses-01 | 60.97 | 81.77 |
| ses-02 | 61.02 | 86.63 |
| ses-03 | 60.08 | 88.90 |

## 5. Reliability checks

- Completed (model, protocol, seed) cells: **30** (target 30 = 3 models × 2 protocols × 5 seeds).
- Within: distinct (subject,session) used = **125** (all 148 used: False); expected rows per (model,seed) = 1480.
  - ⚠️ incomplete within cells: deepconvnet/seed0(1250/1480), deepconvnet/seed1(1250/1480), deepconvnet/seed2(1250/1480), deepconvnet/seed3(1250/1480), deepconvnet/seed4(1250/1480), eegnet/seed0(1250/1480), eegnet/seed1(1250/1480), eegnet/seed2(1250/1480), eegnet/seed3(1250/1480), eegnet/seed4(1250/1480), fbcnet/seed0(1250/1480), fbcnet/seed1(1250/1480), fbcnet/seed2(1250/1480), fbcnet/seed3(1250/1480), fbcnet/seed4(1250/1480)
- Cross: distinct directed pairs = **500** (expected 288); invalid same-session pairs = 0 (must be 0 — no leakage).
  - ⚠️ incomplete cross cells: deepconvnet/seed0(500/288), deepconvnet/seed1(500/288), deepconvnet/seed2(500/288), deepconvnet/seed3(500/288), deepconvnet/seed4(500/288), eegnet/seed0(500/288), eegnet/seed1(500/288), eegnet/seed2(500/288), eegnet/seed3(500/288), eegnet/seed4(500/288), fbcnet/seed0(500/288), fbcnet/seed1(500/288), fbcnet/seed2(500/288), fbcnet/seed3(500/288), fbcnet/seed4(500/288)
- NaN metric cells: none.
- Leakage: by construction within=disjoint folds (val carved from train only), cross=different sessions; no test trials in train/val.

## 6. Figures

- `within_session_accuracy_boxplot.png`
- `cross_session_accuracy_matrix_by_model.png`
- `protocol_comparison.png`

## Notes

- Mainline = the THREE baseline architectures. CAP-EEGNet (v1/v2) and all agent/toolkit/prototype/confidence/online/fine-tuning modules remain FUTURE work.
- LOSO and 41/10 are not run.
