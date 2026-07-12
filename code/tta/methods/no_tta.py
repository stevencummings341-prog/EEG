"""No-TTA baseline: reuse frozen model predictions from FeatureBundle."""

from __future__ import annotations

import numpy as np

from code.tta.feature_sources.base import FeatureBundle
from code.tta.methods.base import MethodResult, TTAMethod


class NoTTAMethod(TTAMethod):
    name = "no_tta"
    uses_target_labels = False

    def run(self, bundle: FeatureBundle, **kwargs) -> MethodResult:
        bundle = self._prepare_label_free(bundle)
        if bundle.target_pred is not None:
            pred = np.asarray(bundle.target_pred, dtype=np.int64).ravel()
        elif bundle.target_logits is not None:
            pred = np.asarray(bundle.target_logits).argmax(axis=1).astype(np.int64)
        elif bundle.target_probs is not None:
            pred = np.asarray(bundle.target_probs).argmax(axis=1).astype(np.int64)
        else:
            raise ValueError(
                f"no_tta requires target_pred/logits/probs on cell {bundle.cell_id}"
            )
        return MethodResult(
            pred=pred,
            method=self.name,
            used_target_labels=False,
            oracle_diagnostic_only=False,
            not_deployable=False,
            geometry="",
            filter_k=None,
            initialization="frozen_model_pred",
        )
