"""Unified FeatureBundle + FeatureSource protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class FeatureBundle:
    """Model-agnostic feature payload for one cross-session cell.

    Target true labels (``target_y_true``) are for **evaluation / Oracle only**.
    Label-free TTA methods must not read them for adaptation (enforced by
    ``code.tta.oracle.label_guard``).
    """

    cell_id: str
    dataset: str
    model: str
    seed: int
    subject: str
    source_session: str
    target_session: str

    source_features: Optional[np.ndarray] = None  # [Ns, D]
    source_labels: Optional[np.ndarray] = None  # [Ns]
    target_features: Optional[np.ndarray] = None  # [Nt, D]
    target_logits: Optional[np.ndarray] = None
    target_probs: Optional[np.ndarray] = None
    target_pred: Optional[np.ndarray] = None
    target_conf: Optional[np.ndarray] = None
    target_y_true: Optional[np.ndarray] = None  # Oracle / eval only

    feature_source: str = "unknown"
    npz_path_resolved: str = ""
    npz_path_original: str = ""
    embedding_dim: Optional[int] = None
    n_source: int = 0
    n_target: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def freeze_for_label_free(self) -> "FeatureBundle":
        """Return a shallow copy with target_y_true cleared (for label-free methods)."""
        return FeatureBundle(
            cell_id=self.cell_id,
            dataset=self.dataset,
            model=self.model,
            seed=self.seed,
            subject=self.subject,
            source_session=self.source_session,
            target_session=self.target_session,
            source_features=self.source_features,
            source_labels=self.source_labels,
            target_features=self.target_features,
            target_logits=self.target_logits,
            target_probs=self.target_probs,
            target_pred=self.target_pred,
            target_conf=self.target_conf,
            target_y_true=None,
            feature_source=self.feature_source,
            npz_path_resolved=self.npz_path_resolved,
            npz_path_original=self.npz_path_original,
            embedding_dim=self.embedding_dim,
            n_source=self.n_source,
            n_target=self.n_target,
            metadata=dict(self.metadata),
        )


def make_cell_id(
    dataset: str,
    model: str,
    seed: int,
    subject: str,
    source_session: str,
    target_session: str,
) -> str:
    """Canonical cell_id (PHASE3_ROUTE_PLAN hard constraint #2)."""
    return (
        f"{dataset}__{model}__seed{int(seed)}__{subject}__"
        f"{source_session}->{target_session}"
    )


class FeatureSource:
    """Protocol-like base for feature providers."""

    name: str = "base"

    def load_cell(self, **kwargs) -> FeatureBundle:
        raise NotImplementedError
