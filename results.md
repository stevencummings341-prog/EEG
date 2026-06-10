---
title: "Results Quick Reference"
tags:
  - "#pipeline/4_analysis"
  - "#modality/eeg"
created: "2026-06-10"
updated: "2026-06-10"
status: "active"
---

# Results Quick Reference

| Stage | Result |
|:---|:---|
| Drift diagnostic | Spatial + spectral drift dominates; amplitude is stable. |
| Baseline | EEGNet 0.807 within vs 0.711 cross; DeepConvNet 0.766 vs 0.681; FBCNet 0.720 vs 0.628. |
| Multi-source | `ses-01+02 -> ses-03` improves over best single source for all three models. |
| Alignment | No-learning alignment is insufficient; BN-stat is only a small positive gain. |
| Prototype Drift | Not run yet. |
