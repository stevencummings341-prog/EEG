---
title: "WBCIC-SHU Phase 3 TTA results"
tags:
  - "#method/test_time_adaptation"
created: "2026-07-10"
updated: "2026-07-12"
status: "pretrained_ready_engineering"
---

# `4_experiments/wbci_shu/tta/`

Phase 3 TTA backend outputs: Round-1 scaffold + Pretrained-Model Readiness Round
(mock live-inference validated, **WBCIC full A0 complete**, SHU smoke elsewhere).

**This is not a full T3A experiment.** Smoke / A0 numbers must not be read as scientific
proof that T3A works or fails. A0 is plumbing consistency vs Phase 2c only.

**Status (honest):** Round-1 scaffold complete · mock live-model path validated ·
WBCIC full A0 complete · real pretrained model not yet integrated · formal Oracle
decision not run · full T3A not run.

```text
tta/
├── smoke/                 # minimal no_tta + t3a_minimal (+ selected_smoke_cells)
├── replay_validation/     # No-TTA vs Phase 2c (smoke + full_a0_* tables)
├── oracle_diagnostic/     # diagnostic-only Oracle (used_target_labels=True)
├── method_catalog/        # candidate list (most not implemented)
├── reports/               # framework + FULL_A0 + pretrained readiness pointer
├── tables/
└── figures/
```

## How to run

```bash
# default = tiny smoke (safe)
python code/run.py --config code/configs/experiments/phase3_tta.yaml --device cpu

# opt-in full A0 (no_tta replay only; not the default)
python code/run.py --config code/configs/experiments/phase3_tta_full_a0.yaml --device cpu
```

## Integration note

**Authoritative contract:**  
[`3_online_adaptation/PRETRAINED_MODEL_INTEGRATION_CONTRACT.md`](../../../3_online_adaptation/PRETRAINED_MODEL_INTEGRATION_CONTRACT.md)

Short pointer: [`reports/PRETRAINED_READYNESS_NOTES.md`](reports/PRETRAINED_READYNESS_NOTES.md)

Future real pretrained models: add `code/tta/adapters/<name>.py` + config
`model_adapter` / checkpoint fields. Do **not** rewrite the TTA backend.
Mock / `eegnet` adapter names in smoke config are fixtures, not a senior research model.
