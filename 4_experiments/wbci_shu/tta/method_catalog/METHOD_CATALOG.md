---
title: "Phase 3 Method Catalog (WBCIC-SHU)"
tags:
  - "#method/test_time_adaptation"
created: "2026-07-10"
updated: "2026-07-10"
status: "catalog"
---

# Method Catalog — WBCIC-SHU TTA

> Round-1: **catalog + documentation only** for most candidates.
> Implemented code: `no_tta`, `t3a_minimal`, `target_label_oracle_proto`.
> This is **not** a full experiment matrix execution.

Machine-readable twin: `code/tta/method_catalog.yaml`.

## Implemented (Round-1)

| id | kind | used_target_labels | notes |
|:---|:---|:---|:---|
| `no_tta` | baseline | False | Replay Phase 2c pred/logits |
| `t3a_minimal` | TTA | False | One clean minimal variant (config-driven) |
| `target_label_oracle_proto` | Oracle diagnostic | True | Not deployable |

Smoke may use `src_proto + cosine + filter_k=20`. That combination is a **smoke example**, not a permanent project default.

## Candidates (not implemented in Round-1)

### TTA / T3A axes
- `t3a_classifier_weight_init`, `t3a_source_proto_init`, `t3a_target_support_only`
- `t3a_cumulative`, `t3a_episodic`
- `confidence_filtering`, `entropy_filtering`, `margin_filtering`
- `top_k_filtering`, `threshold_filtering`
- `cosine_geometry`, `dot_product_geometry`, `euclidean_geometry`
- `class_balanced_support`, `reliability_weighted_support`
- `shrinkage_prototype`, `source_target_prototype_interpolation`

### Oracle candidates (diagnostic only if implemented later)
- `cosine_oracle_proto`, `shrinkage_oracle`, `reliability_weighted_oracle`
- `scatter_aware_oracle`, `source_target_interpolation_oracle`
- `mahalanobis_covariance_aware_oracle` (needs shrinkage+ridge+cond-number)

## Binary-class note

For K=2 MI, prediction **entropy ranking ≡ max-confidence ranking**. Do not treat entropy vs max-conf as a real ablation on this task.

## Provisional Oracle notes

> Current Oracle thresholds (+3pp / +1pp) are provisional and should be revisited after pretrained model integration.
