"""tests for no_tta method."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.tta.eval.metrics import build_result_row
from code.tta.eval.schema import validate_result_row
from code.tta.feature_sources.base import FeatureBundle
from code.tta.methods.no_tta import NoTTAMethod
from code.tta.oracle.label_guard import run_label_free


def _bundle():
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    pred = np.array([0, 1, 1, 1], dtype=np.int64)
    return FeatureBundle(
        cell_id="ds__m__seed0__sub__ses-01->ses-02",
        dataset="ds",
        model="m",
        seed=0,
        subject="sub",
        source_session="ses-01",
        target_session="ses-02",
        target_pred=pred,
        target_y_true=y,
        n_target=4,
        feature_source="test",
    )


def test_no_tta_acc_and_schema():
    b = _bundle()
    res = run_label_free(NoTTAMethod(), b)
    assert list(res.pred) == [0, 1, 1, 1]
    assert res.used_target_labels is False
    row = build_result_row(bundle=b, result=res, model_adapter="embedding_only")
    missing = validate_result_row(row)
    assert missing == []
    assert abs(row["acc"] - 0.75) < 1e-9
