---
title: "Minimal T3A Smoke Report"
created: "2026-07-12"
status: "smoke"
---

# Minimal T3A Smoke Report

**Pipeline status:** passed

This smoke only checks that the **pipeline can run** (no_tta + one minimal T3A variant).
**Do not** interpret accuracy deltas as scientific evidence that T3A works or fails.

- selected subjects: sub-002, sub-017
- result rows: 2

## Notes

- Round-1 scaffold smoke only; not a scientific conclusion.
- t3a_minimal: skipped (run_t3a=false; no_tta only)
- Binary MI: entropy ranking ≡ max-confidence ranking (K=2).

## Known limitations

- Single minimal T3A variant; not a full ablation.
- EEGNet appears only as an example adapter name in smoke config.
- Pretrained model not integrated.
