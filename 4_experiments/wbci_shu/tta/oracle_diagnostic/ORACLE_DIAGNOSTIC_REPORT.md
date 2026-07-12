---
title: "Oracle Diagnostic Report (minimal)"
created: "2026-07-12"
status: "diagnostic_only"
---

# Oracle Diagnostic Report (minimal)

**Status:** minimal_oracle_ran

Oracle methods use target true labels and are **diagnostic only** (`used_target_labels=True`, `oracle_diagnostic_only=True`, `not_deployable=True`). They are **not** deployable TTA methods.

- result rows: 3

> Current Oracle thresholds (+3pp / +1pp) are provisional and should be revisited after pretrained model integration.

## Notes

- Only target_label_oracle_proto implemented in Round-1.
- Current Oracle thresholds (+3pp / +1pp) are provisional and should be revisited after pretrained model integration.

Complex Oracle candidates (shrinkage / reliability / mahalanobis / …) are catalog-only in Round-1.
