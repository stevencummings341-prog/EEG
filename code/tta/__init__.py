"""Model-agnostic Test-Time Adaptation (TTA) backend scaffold (Phase 3 Round-1).

This package is intentionally **not** bound to EEGNet / DeepConvNet / FBCNet.
Those models exist only as optional example adapters under ``tta.adapters``.

Layers:
  * adapters        — optional ModelAdapter protocol + registry
  * feature_sources — embedding replay / (scaffold) live model inference
  * methods         — no_tta + minimal T3A (+ catalog for future variants)
  * oracle          — diagnostic-only target-label methods + label guard
  * eval / report   — result schema + smoke reporters

Round-1 status: scaffold + smoke runnable. Not a full T3A experiment.
Pretrained models are not integrated yet; add an adapter + config later.
"""

from __future__ import annotations

from code.tta.exceptions import UnsupportedAdapterFeature

__all__ = [
    "UnsupportedAdapterFeature",
]