---
title: "code/tta — model-agnostic TTA backend"
tags:
  - "#method/test_time_adaptation"
created: "2026-07-12"
updated: "2026-07-12"
status: "pretrained_ready_engineering"
---

# `code/tta/`

Phase 3 **model-agnostic TTA backend** (Round-1 scaffold + Pretrained-Model Readiness).

**Not** a full T3A experiment. **Real pretrained model not yet integrated.**
Mock live-inference path is validated with test fixtures only (`tests/tta/support/`).

## Layout

| Subpackage | Role |
|:---|:---|
| `adapters/` | `ModelAdapter` + `AdapterCapabilities` + registry (`embedding_only`, example `baseline_torch` / EEGNet aliases) |
| `feature_sources/` | `EmbeddingReplaySource` (default runner); `ModelInferenceSource` (real checkpoint→forward path) |
| `methods/` | `no_tta`, `t3a_minimal` |
| `oracle/` | Diagnostic-only target-label methods + label guard (`run_label_free` strips labels) |
| `eval/` / `report/` | Result schema + smoke / full-A0 reporters |
| `method_catalog.yaml` | Candidates (most not implemented) |

## Pretrained integration

Authoritative contract:

[`3_online_adaptation/PRETRAINED_MODEL_INTEGRATION_CONTRACT.md`](../../3_online_adaptation/PRETRAINED_MODEL_INTEGRATION_CONTRACT.md)

Summary: implement a thin adapter → register → config checkpoint → preflight → smoke → A0 consistency → minimal T3A → then Phase 3B Oracle. Do not hard-bind senior architectures into method code.

## Status phrases

- Round-1 scaffold complete
- mock live-model path validated
- WBCIC full A0 complete / SHU replay smoke passed (see `4_experiments/*/tta/`)
- real pretrained model not yet integrated
- formal Oracle decision not run
- full T3A not run
