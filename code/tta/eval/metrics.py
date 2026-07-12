"""Metrics helpers for TTA evaluation rows."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from code.experiments.metrics import classification_metrics
from code.tta.eval.schema import empty_result_row
from code.tta.feature_sources.base import FeatureBundle
from code.tta.methods.base import MethodResult


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if len(y_true) == 0:
        return {"acc": float("nan"), "balanced_acc": float("nan")}
    m = classification_metrics(y_true, y_pred)
    return {
        "acc": float(m["accuracy"]),
        "balanced_acc": float(m["balanced_accuracy"]),
    }


def build_result_row(
    *,
    bundle: FeatureBundle,
    result: MethodResult,
    model_adapter: str,
    no_tta_acc: Optional[float] = None,
    failure_reason: str = "",
) -> Dict[str, Any]:
    y_true = bundle.target_y_true
    metrics = {"acc": float("nan"), "balanced_acc": float("nan")}
    if y_true is not None and result.pred is not None and len(result.pred) == len(y_true):
        metrics = evaluate_predictions(y_true, result.pred)

    delta = float("nan")
    neg = False
    if no_tta_acc is not None and np.isfinite(metrics["acc"]) and np.isfinite(no_tta_acc):
        delta = float(metrics["acc"] - no_tta_acc)
        neg = bool(delta < 0)

    return empty_result_row(
        dataset=bundle.dataset,
        model_adapter=model_adapter,
        method=result.method,
        seed=int(bundle.seed),
        subject=bundle.subject,
        source_session=bundle.source_session,
        target_session=bundle.target_session,
        cell_id=bundle.cell_id,
        feature_source=bundle.feature_source,
        n_source=int(bundle.n_source),
        n_target=int(bundle.n_target),
        acc=metrics["acc"],
        balanced_acc=metrics["balanced_acc"],
        delta_vs_no_tta=delta,
        negative_transfer=neg,
        used_target_labels=bool(result.used_target_labels),
        oracle_diagnostic_only=bool(result.oracle_diagnostic_only),
        not_deployable=bool(result.not_deployable),
        geometry=result.geometry or "",
        filter_k="" if result.filter_k is None else int(result.filter_k),
        initialization=result.initialization or "",
        npz_path_resolved=bundle.npz_path_resolved,
        failure_reason=failure_reason or result.failure_reason or "",
    )
