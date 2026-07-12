"""TTA method interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from code.tta.feature_sources.base import FeatureBundle


@dataclass
class MethodResult:
    pred: np.ndarray
    method: str
    used_target_labels: bool = False
    oracle_diagnostic_only: bool = False
    not_deployable: bool = False
    geometry: str = ""
    filter_k: Optional[int] = None
    initialization: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)
    failure_reason: str = ""


class TTAMethod(ABC):
    """Label-free by default. Subclasses must not use target_y_true for adapt."""

    name: str = "base"
    uses_target_labels: bool = False

    @abstractmethod
    def run(self, bundle: FeatureBundle, **kwargs) -> MethodResult:
        ...

    def _prepare_label_free(self, bundle: FeatureBundle) -> FeatureBundle:
        """Strip target_y_true so accidental reads fail closed.

        Stronger leakage checks live in ``code.tta.oracle.label_guard``.
        """
        if self.uses_target_labels:
            return bundle
        return bundle.freeze_for_label_free()
