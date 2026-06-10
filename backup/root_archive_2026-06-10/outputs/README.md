# outputs/

Generated outputs for the EEG MI cross-session drift project.

## Main layout

```text
outputs/
├── analysis/
│   └── session_drift_v1/      # drift diagnosis from preprocessed data (no model training)
└── experiments/
    └── baseline_v1/           # completed static baseline (within + cross)
```

## What to read

- Drift diagnosis: `analysis/session_drift_v1/README.md`
- Static baseline: `experiments/baseline_v1/BASELINE_REPORT.md`
- All baseline figures: `experiments/baseline_v1/figures/`

## Notes

- `baseline_v1/provenance/` keeps the original raw run folders for reproducibility.
- Step 2 no-learning adaptation has not been run yet.
- Raw/workspace2 data is not stored here.
