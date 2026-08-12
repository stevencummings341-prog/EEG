"""Published baselines from the DSGNet paper's Table II, from their official repositories."""

from .adapter import (
    BASELINE_NAMES,
    PaperBaselineClassifier,
    PaperBaselineConfig,
    build_paper_baseline,
)

__all__ = [
    "BASELINE_NAMES",
    "PaperBaselineClassifier",
    "PaperBaselineConfig",
    "build_paper_baseline",
]
