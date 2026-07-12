"""tests for Oracle label guard."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.tta.exceptions import LabelLeakageError
from code.tta.feature_sources.base import FeatureBundle
from code.tta.methods.no_tta import NoTTAMethod
from code.tta.oracle.label_guard import run_label_free, run_oracle
from code.tta.oracle.target_label_proto import TargetLabelProtoOracle


def _bundle():
    rng = np.random.RandomState(0)
    z = rng.randn(20, 6).astype(np.float32)
    y = np.array([0] * 10 + [1] * 10, dtype=np.int64)
    z[10:] += 1.5
    return FeatureBundle(
        cell_id="oracle_cell",
        dataset="t",
        model="m",
        seed=0,
        subject="s",
        source_session="ses-01",
        target_session="ses-02",
        target_features=z,
        target_y_true=y,
        target_pred=y.copy(),
        n_target=20,
        feature_source="synth",
    )


def test_label_free_cannot_pass_y_true():
    b = _bundle()
    with pytest.raises(LabelLeakageError):
        run_label_free(NoTTAMethod(), b, use_target_labels=True)


def test_oracle_flags_required():
    b = _bundle()
    res = run_oracle(TargetLabelProtoOracle(geometry="cosine"), b)
    assert res.used_target_labels is True
    assert res.oracle_diagnostic_only is True
    assert res.not_deployable is True
    assert res.pred.shape == (20,)
