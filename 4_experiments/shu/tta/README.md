---
title: "SHU Phase 3 TTA results"
tags:
  - "#method/test_time_adaptation"
created: "2026-07-10"
updated: "2026-07-12"
status: "replay_smoke_passed"
---

# `4_experiments/shu/tta/`

SHU-scoped Phase 3 TTA outputs (dataset-parallel to WBCIC).

**SHU replay smoke passed** (2026-07-12): eegnet / seed0 / 2 cells / **no_tta only**;
paths under `outputs/experiments/shu/` (not WBCIC); `|Δ|<1e-6` vs Phase 2c. This is
**route validation only** — not a scientific SHU TTA conclusion. SHU remains
**single-column** (near-chance); do not merge into the main WBCIC×{EEGNet,DeepConvNet}
conclusion.

```bash
python code/run.py --config code/configs/experiments/shu_phase3_tta.yaml --device cpu
```

Not a full T3A experiment. Oracle skipped in SHU smoke (`run_oracle: false`).
**Real pretrained model not yet integrated.** Formal Oracle decision not run; full T3A not run.

**Authoritative integration contract:**  
[`3_online_adaptation/PRETRAINED_MODEL_INTEGRATION_CONTRACT.md`](../../../3_online_adaptation/PRETRAINED_MODEL_INTEGRATION_CONTRACT.md)
