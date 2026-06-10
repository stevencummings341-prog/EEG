# alignment_baseline_v1 — Step 2 no-learning adaptation baseline

Unsupervised **test-time alignment** baseline (NO target labels, NO weight learning
on target). The model trains only on the source session(s); the target session is
used only through its UNLABELED X for alignment statistics; `y_test` is used only
for the final evaluation. No `optimizer.step` on the target (only BatchNorm
running-stat updates for `bn_statistics_adaptation`).

## Methods
- `none_reference` — no alignment; **referenced** from baseline_v1 cross rows (not re-run).
- `session_zscore` — per-channel mean/std normalization.
- `euclidean_alignment` — whiten by `R^{-1/2}`, R = arithmetic mean of trial covariances
  (eigh inverse-sqrt; eps ridge + diagonal shrinkage). Matrix is 58×58.
- `riemannian_alignment` — whiten by `G^{-1/2}`, G = **log-Euclidean** SPD mean
  `expm(mean_i logm(C_i))` (numpy/scipy only; pyriemann is detected but never required).
- `bn_statistics_adaptation` — train on source, then forward unlabeled target X to
  refresh BatchNorm running mean/var only (no loss/backward/optimizer).
- `filterbank_reweighting` — θ/μ/β/low-γ FIR sub-bands; reweight each band (scalar gain,
  clipped) so the target band-power profile matches the source profile.

## Protocols
- single-source directed pairs (ses-i → ses-j, both ok).
- multi-source ses-01+ses-02 → ses-03 (all three ok).

## Models / seeds
- EEGNet / DeepConvNet / FBCNet; seeds 0–4.

## Layout
```
alignment_baseline_v1/
├── README.md
├── ALIGNMENT_BASELINE_REPORT.md      # filled by summarizer
├── RUN_STATUS.md                     # filled by summarizer
├── manifest_sources.json
├── resolved_config_summary.yaml
├── full_job_ids.txt                  # training job ids (E)
└── cross_session/
    ├── runs/      # per-(method,model,scope,seed) result CSVs (+ meta json)
    ├── splits/    # per-(task,seed) split JSON (train/val idx, sessions)
    ├── tables/    # results_alignment_all.csv + aggregates (summarizer)
    └── figures/   # comparison/gain figures (summarizer)
```

Do NOT overwrite `baseline_v1/`. `none_reference` is read from
`outputs/experiments/baseline_v1/cross_session/tables/results_cross_session_all.csv`.
