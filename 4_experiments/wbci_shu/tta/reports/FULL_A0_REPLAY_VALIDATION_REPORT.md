---
title: "Full A0 No-TTA Replay Validation Report"
created: "2026-07-12"
status: "diagnostic"
---

# Full A0 No-TTA Replay Validation

**dataset:** wbci_shu
**verdict:** complete

Validates that offline embedding replay of **all canonical Phase 2c cells** reproduces `acc_target` exactly under `no_tta` (frozen-model predictions already stored in the npz). This is a plumbing/consistency check, **not** a T3A/TTA effectiveness claim.

## Canonical universe

- metrics cells (label_based + euclidean, requested models/seeds): 4320
- index cells (embed_index__*.csv): 4320
- embedding files (npz) found: 4320
- canonical valid cells (metrics ∩ embeddings): 4320
- metrics duplicate cell_ids: 0
- index duplicate cell_ids (dup split rows): 0
- index cells with incomplete/unexpected split set: 0
- embedding duplicate cell_ids: 0
- missing embeddings (in metrics, no npz found): 0
- unexpected embeddings (npz found, not in metrics): 0
- missing from index (in metrics, absent from embed_index): 0
- unexpected in index (in embed_index, absent from metrics): 0

## Replay results

- cells attempted: 4320
- cells load/eval failed: 0
- cells replayed successfully: 4320
- passed \|Δ\| < 1e-06: 4320
- failed tolerance: 0
- max \|Δ\|: 0.0
- wall time (s): 756.4

## Max |Δ| by model / seed / direction (worst 20)

| model | seed | direction | n | max_abs_delta |
|---|---|---|---|---|
| deepconvnet | 0 | ses-01->ses-02 | 47 | 0.0 |
| deepconvnet | 0 | ses-01->ses-03 | 48 | 0.0 |
| deepconvnet | 0 | ses-02->ses-01 | 47 | 0.0 |
| deepconvnet | 0 | ses-02->ses-03 | 49 | 0.0 |
| deepconvnet | 0 | ses-03->ses-01 | 48 | 0.0 |
| deepconvnet | 0 | ses-03->ses-02 | 49 | 0.0 |
| deepconvnet | 1 | ses-01->ses-02 | 47 | 0.0 |
| deepconvnet | 1 | ses-01->ses-03 | 48 | 0.0 |
| deepconvnet | 1 | ses-02->ses-01 | 47 | 0.0 |
| deepconvnet | 1 | ses-02->ses-03 | 49 | 0.0 |
| deepconvnet | 1 | ses-03->ses-01 | 48 | 0.0 |
| deepconvnet | 1 | ses-03->ses-02 | 49 | 0.0 |
| deepconvnet | 2 | ses-01->ses-02 | 47 | 0.0 |
| deepconvnet | 2 | ses-01->ses-03 | 48 | 0.0 |
| deepconvnet | 2 | ses-02->ses-01 | 47 | 0.0 |
| deepconvnet | 2 | ses-02->ses-03 | 49 | 0.0 |
| deepconvnet | 2 | ses-03->ses-01 | 48 | 0.0 |
| deepconvnet | 2 | ses-03->ses-02 | 49 | 0.0 |
| deepconvnet | 3 | ses-01->ses-02 | 47 | 0.0 |
| deepconvnet | 3 | ses-01->ses-03 | 48 | 0.0 |

## Notes

- models=['eegnet', 'deepconvnet', 'fbcnet'], seeds=[0, 1, 2, 3, 4], tolerance=1e-06
- no_tta replay only; T3A/Tent/SHOT/Oracle NOT run in full_a0_replay mode.
- index files used: 15

If verdict is `partial` or `blocked`, see the accompanying `full_a0_universe_consistency.csv` and `full_a0_replay_delta_table.csv` for the exact mismatch/failure rows. Do **not** expand T3A/Oracle matrices until this is `complete`.
