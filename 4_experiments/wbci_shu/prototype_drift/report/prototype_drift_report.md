---
title: "Phase 2c Prototype Drift Analysis Report"
tags:
  - "#pipeline/5_dl_model"
  - "#modality/eeg"
  - "#method/domain_generalization"
  - "#paradigm/motor_imagery"
created: "2026-06-11"
updated: "2026-06-11"
status: "active"
---

# Phase 2c -- Prototype Drift Analysis

> Frozen-model diagnostic: does cross-session accuracy drop come from class-prototype drift in the penultimate embedding space? (EEGNet / DeepConvNet / FBCNet, WBCIC-SHU 2C)

**target labels are used only for offline diagnostic analysis, not for training or adaptation.**

## 1. Core conclusion (honest)

- Analyzed 4320 (subject x direction x model x seed) frozen-model cells (canonical subset prototype=label_based, distance=euclidean).
- prototype_drift_mean vs acc_drop: Spearman rho=0.352 (p=1.64e-126, n=4320) -> moderate positive (significant).
- prototype_direction_cosine vs acc_drop: rho=-0.237 (p=2.9e-56) -> weak negative (significant).
- target_negative_margin_rate vs acc_drop: rho=0.313 (p=1.46e-98) -> moderate positive (significant).
- separation_change vs acc_drop: rho=0.389 (p=1.82e-156) -> moderate positive (significant).
- fisher_change vs acc_drop: rho=0.359 (p=7.72e-132) -> moderate positive (significant).
- Target-label-in-training leakage detected: False (must be False).

## 2. Goal

Test whether the cross-session decode drop is explained by drift of the task representation (per-class prototypes) in the model's penultimate embedding space, and which embedding-geometry signal (raw drift, direction consistency, margin, separation collapse, within-class scatter, Fisher ratio) best predicts the drop. This is a diagnostic, not an adaptation method.

## 3. Why this experiment follows Phase 2b

Phase 1 found a ~10pp cross-session drop. Phase 2b showed no-learning statistical alignment (z-score / EA / Riemannian / BN-stats / filterbank) is insufficient (only BN-stats gave a small positive gain, none reached +2pp, and high-drift subjects were helped least). That points beyond mean/variance/covariance shift toward task-representation reorganization -- which this experiment measures directly in embedding space.

## 4. Protocol

- Dataset: WBCIC-SHU 2C (left vs right MI), status=ok sessions only, run_id=prototype_drift_v1.
- For each subject x directed (source->target) session pair x model x seed: train on SOURCE session only (train + a stratified val slice from source train for early stopping); TARGET session is test-only.
- Subjects with all 3 ok sessions contribute 6 directed pairs; subjects with 2 ok sessions contribute their available directed pairs; subjects with <2 ok sessions are skipped (consistent with the Phase 1 cross-session protocol).
- Seeds: 0,1,2,3,4. Training recipe identical to Phase 1 baseline (Adam, lr=1e-3, batch=16, max_epochs=100, early-stopping patience=20, val_fraction=0.2).
- The model is frozen after training; embeddings are extracted with no gradient.

## 5. Model and embedding extraction

- Three baseline architectures share one trainer and one forward contract `{logits, features, confidence}`.
- Main embedding = penultimate `features` (flatten before the linear head): EEGNet/DeepConvNet/FBCNet expose this directly.
- Auxiliary signals saved per trial: logits, softmax probability, prediction, confidence.
- None of the three baselines has a learned confidence head, so confidence falls back to the max softmax probability (documented; not a learned calibration head).

## 6. Prototype definition

Per class, computed separately on SOURCE-train and TARGET-test embeddings:
- `label_based`: mean(z_i | y_i = class).
- `confidence_weighted`: confidence-weighted mean of z_i within class.
- `correct_only`: mean(z_i | y_i = class AND prediction_i = class).
Distances: euclidean and cosine. (Canonical subset for headline numbers: label_based / euclidean.)

## 7. Leakage control

- Source train/val and target test come from different sessions (asserted).
- Target labels never enter the training loop, optimizer, or early stopping (n_target_labels_used_for_training is 0 on every row; used_target_labels_for_training=False).
- Target X is used only for prediction / embedding extraction; target y is used only for offline prototype/metric diagnostics.
- Verified across all rows: any target-label-in-training leakage = False.

## 8. Main results

Mean cross-session accuracy drop (source_val - target) per model:

| model | mean acc_drop | std | n_cells |
|:---|---:|---:|---:|
| deepconvnet | 0.1209 | 0.1387 | 1440 |
| eegnet | 0.1063 | 0.1478 | 1440 |
| fbcnet | 0.1161 | 0.1193 | 1440 |

Total frozen-model cells analyzed: 4320. Run status counts: {'ok': 4320}.

## 9. Correlation analysis

Canonical subset (ALL models, prototype=label_based, distance=euclidean), each vs acc_drop:

| relationship | n | Pearson r | Spearman rho | Spearman p | r2 | verdict |
|:---|---:|---:|---:|---:|---:|:---|
| Prototype drift (mean) vs acc_drop | 4320 | 0.067 | 0.352 | 1.64e-126 | 0.005 | moderate positive (significant) |
| Prototype direction cosine vs acc_drop | 4320 | -0.264 | -0.237 | 2.9e-56 | 0.070 | weak negative (significant) |
| Class separation change vs acc_drop | 4320 | 0.036 | 0.389 | 1.82e-156 | 0.001 | moderate positive (significant) |
| Target negative-margin rate vs acc_drop | 4320 | 0.365 | 0.313 | 1.46e-98 | 0.133 | moderate positive (significant) |
| Target margin mean vs acc_drop | 4320 | -0.176 | -0.159 | 7.14e-26 | 0.031 | weak negative (significant) |
| Fisher change vs acc_drop | 4320 | 0.247 | 0.359 | 7.72e-132 | 0.061 | moderate positive (significant) |

Full per-(model x prototype_type x distance) breakdown: `tables/prototype_accuracy_correlation.csv`.

## 10. By-model robustness: EEGNet / DeepConvNet / FBCNet

| model | drift_mean rho | direction_cosine rho | neg_margin rho | separation_change rho | fisher_change rho |
|:---|---:|---:|---:|---:|---:|
| deepconvnet | 0.487 | -0.279 | 0.383 | 0.551 | 0.412 |
| eegnet | 0.558 | -0.364 | 0.411 | 0.275 | 0.409 |
| fbcnet | 0.282 | -0.036 | 0.095 | 0.344 | 0.293 |

If the sign/strength differs across models, the prototype-drift story is model-dependent and must not be over-generalized.

## 11. Interpretation

- Does prototype drift explain cross-session drop? raw drift_mean rho=0.352 (moderate positive (significant)).
- Is direction cosine more predictive than raw drift distance? |rho_dir|=0.237 vs |rho_drift|=0.352.
- Does target negative-margin rate explain failures? rho=0.313 (moderate positive (significant)).
- Does class separation collapse occur / predict drop? separation_change rho=0.389 (moderate positive (significant)).
- Does within-class scatter increase / Fisher ratio drop? fisher_change rho=0.359 (moderate positive (significant)).

## 12. Relationship to previous phases

- Phase 0: drift is spatial + spectral, not amplitude.
- Phase 1: ~10pp cross-session drop (EEGNet 0.807->0.711).
- Phase 2a: multi-source helps but does not close the gap.
- Phase 2b: statistical alignment insufficient; high-drift subjects helped least.
- Phase 2c (this): quantifies whether the drop is an embedding-space prototype-drift phenomenon, which would (or would not) justify prototype-based adaptation.

## 13. Limitations

- Diagnostic only; no adaptation is performed or claimed.
- Target prototypes use target labels for offline analysis only.
- Prototypes summarize each class by a single centroid; multi-modal class structure is not captured.
- Confidence is fallback (max softmax), not a learned/calibrated head.
- 2C WBCIC-SHU only; SHU 2022 and cross-dataset are out of scope here.

## 14. Next step

- If prototype drift (and especially direction cosine / negative-margin rate) explains the drop: prototype-based adaptation (Oracle -> few-shot -> pseudo-label) is justified.
- If not: investigate decision-boundary drift, within-class scatter growth, class-separation collapse, or reliability/engagement drift before committing to a prototype adaptation method.

## 15. File list

- `tables/prototype_drift_metrics.csv` -- main per-cell metric table.
- `tables/prototype_table.csv` -- per-class prototype metadata (vectors in embedding npz).
- `tables/prototype_accuracy_correlation.csv` -- correlation analysis.
- `tables/trial_embeddings_index.csv` -- trial-level embedding index (npz references).
- `tables/run_status.csv` + `report/RUN_STATUS.md` -- per-cell run status.
- `figures/drift_vs_acc_drop.png`
- `figures/direction_cosine_vs_acc_drop.png`
- `figures/negative_margin_vs_acc_drop.png`
- `figures/separation_change_vs_acc_drop.png`
- `figures/fisher_change_vs_acc_drop.png`
- `figures/acc_drop_by_model.png`
- `figures/correlation_summary.png`
- Heavy artifacts: `outputs/experiments/prototype_drift_v1/embeddings/` (npz), `checkpoints/prototype_drift_v1/`.
