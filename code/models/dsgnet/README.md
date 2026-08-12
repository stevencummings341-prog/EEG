---
title: "DSGNet — upstream preview (incomplete release)"
tags:
  - "#pipeline/5_dl_model"
  - "#modality/eeg"
created: "2026-08-04"
updated: "2026-08-04"
status: "integration_pending"
---

# DSGNet (Lou et al., IEEE JBHI 2026)

Upstream: <https://github.com/xicheng105/DSGNet>

> README quote: *The full code will be released upon acceptance of the paper.
> This repository currently includes model architecture and essential components
> for reproducibility preview.*

## Files here

| File | Source |
|:---|:---|
| `_model_raw.py` | upstream `temp_model.py` (includes `DSGNet`) |
| `_modules_raw.py` | upstream `temp_module.py` (includes `ClassAlignmentLoss`) |
| `UPSTREAM_README.md` | upstream README |

## Status in this project

- **Not yet** registered in `code.models.registry` / experiment configs.
- Hard deps in raw files: `docstring_inheritance`, `torchinfo`, `mne`, `einops`,
  relative `from modules import ...` — need a cleaned adapter before training.
- Forward returns `(logits, dist_features)` and expects `data_list` (multi-domain);
  must wrap to `{logits, features, confidence}` + optional `training_step` for
  CE + `ClassAlignmentLoss` (guard single-domain / zero inter-domain divisor).

## Advisor ask

Confirm whether a full training script + hyperparams (batching by subject domains,
loss weights, epochs) will be provided, or whether we should reconstruct training
from the paper alone using this architecture preview.
