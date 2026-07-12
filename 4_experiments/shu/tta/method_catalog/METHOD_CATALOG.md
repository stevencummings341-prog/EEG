---
title: "Phase 3 Method Catalog (SHU)"
tags:
  - "#method/test_time_adaptation"
created: "2026-07-10"
updated: "2026-07-10"
status: "catalog"
---

# Method Catalog — SHU TTA

> Same Round-1 catalog as WBCIC. SHU results are **single-column / external validation**
> (near-chance floor); do not merge into the main WBCIC conclusion.
>
> Implemented: `no_tta`, `t3a_minimal`, `target_label_oracle_proto` only.
> Other candidates are registered here, **not implemented** in Round-1.

See also: `code/tta/method_catalog.yaml`, `4_experiments/wbci_shu/tta/method_catalog/METHOD_CATALOG.md`.

## Implemented (Round-1)

| id | kind | used_target_labels |
|:---|:---|:---|
| `no_tta` | baseline | False |
| `t3a_minimal` | TTA | False |
| `target_label_oracle_proto` | Oracle diagnostic | True |

## Candidates (not implemented)

TTA: classifier-weight init, source/target support variants, cumulative/episodic,
confidence/entropy/margin/top-k/threshold filters, cosine/dot/euclidean geometries,
class-balanced / reliability-weighted support, shrinkage, source–target interpolation.

Oracle: cosine / shrinkage / reliability / scatter-aware / interpolation / mahalanobis.

## Notes

- Binary MI: entropy ≡ max-confidence ranking (K=2).
- Oracle thresholds (+3pp / +1pp) are **provisional**.
- Pretrained model not integrated yet — add adapter/config later.
