"""Embedding-only adapter — no live model forward.

Used when Phase 2c npz embeddings are the sole feature source. Any attempt to
run a network forward raises ``UnsupportedAdapterFeature`` clearly.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from code.tta.adapters.base import ModelAdapter
from code.tta.exceptions import UnsupportedAdapterFeature


class EmbeddingOnlyAdapter(ModelAdapter):
    name = "embedding_only"

    def __init__(
        self,
        *,
        feature_dim: Optional[int] = None,
        n_classes: int = 2,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._feature_dim = feature_dim
        self._n_classes = int(n_classes)
        self._metadata = dict(metadata or {})
        self._metadata.setdefault("adapter_name", self.name)
        self._metadata.setdefault("n_classes", self._n_classes)
        if feature_dim is not None:
            self._metadata.setdefault("feature_dim", int(feature_dim))

    def forward_features(self, x: Any):
        raise UnsupportedAdapterFeature(
            "EmbeddingOnlyAdapter has no live model; use FeatureSource replay."
        )

    def forward_logits(self, x: Any):
        raise UnsupportedAdapterFeature(
            "EmbeddingOnlyAdapter has no live model; use FeatureSource replay."
        )

    def predict_proba(self, x: Any):
        raise UnsupportedAdapterFeature(
            "EmbeddingOnlyAdapter has no live model; use FeatureSource replay."
        )

    def get_classifier_weights(self):
        raise UnsupportedAdapterFeature(
            "EmbeddingOnlyAdapter has no classifier weights."
        )

    def get_source_prototypes(self):
        raise UnsupportedAdapterFeature(
            "EmbeddingOnlyAdapter has no stored source prototypes; "
            "compute them from FeatureBundle.source_features if available."
        )

    def get_feature_dim(self) -> int:
        if self._feature_dim is None:
            raise UnsupportedAdapterFeature(
                "EmbeddingOnlyAdapter feature_dim unknown until embeddings are loaded."
            )
        return int(self._feature_dim)

    def get_model_metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

    def load_checkpoint(self, path: str) -> None:
        raise UnsupportedAdapterFeature(
            "EmbeddingOnlyAdapter does not load checkpoints."
        )
