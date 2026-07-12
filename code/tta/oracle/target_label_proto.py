"""Minimal target-label prototype Oracle (diagnostic only).

Uses target true labels to build class prototypes, then classifies target
features by similarity. Not deployable.
"""

from __future__ import annotations

import numpy as np

from code.tta.feature_sources.base import FeatureBundle
from code.tta.methods.base import MethodResult
from code.tta.methods.t3a_minimal import _similarity
from code.tta.oracle.base import OracleMethod


class TargetLabelProtoOracle(OracleMethod):
    name = "target_label_oracle_proto"

    def __init__(
        self,
        *,
        geometry: str = "cosine",
        n_classes: int = 2,
    ) -> None:
        self.geometry = geometry
        self.n_classes = int(n_classes)

    def run(self, bundle: FeatureBundle, **kwargs) -> MethodResult:
        if bundle.target_features is None or bundle.target_y_true is None:
            raise ValueError(
                f"{bundle.cell_id}: target_label_oracle needs features + y_true"
            )
        z = np.asarray(bundle.target_features, dtype=np.float32)
        y = np.asarray(bundle.target_y_true, dtype=np.int64).ravel()
        dim = z.shape[1]
        protos = np.zeros((self.n_classes, dim), dtype=np.float32)
        for c in range(self.n_classes):
            mask = y == c
            if mask.any():
                protos[c] = z[mask].mean(axis=0)
            # else leave zeros (empty-class fallback)
        scores = _similarity(z, protos, self.geometry)
        pred = scores.argmax(axis=1).astype(np.int64)
        return self._tagged(
            MethodResult(
                pred=pred,
                method=self.name,
                used_target_labels=True,
                oracle_diagnostic_only=True,
                not_deployable=True,
                geometry=self.geometry,
                filter_k=None,
                initialization="target_label_proto",
                extras={"diagnostic_only": True},
            )
        )
