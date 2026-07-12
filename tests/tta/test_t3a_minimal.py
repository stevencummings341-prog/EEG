"""tests for minimal T3A."""

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
from code.tta.methods.t3a_minimal import MinimalT3AMethod
from code.tta.oracle.label_guard import run_label_free


def _synth_bundle(n: int = 40, d: int = 8):
    rng = np.random.RandomState(0)
    # separable-ish source
    src_z = np.vstack(
        [
            rng.randn(n // 2, d).astype(np.float32) + np.array([2.0] + [0] * (d - 1)),
            rng.randn(n // 2, d).astype(np.float32) - np.array([2.0] + [0] * (d - 1)),
        ]
    )
    src_y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int64)
    tgt_z = src_z + 0.1 * rng.randn(n, d).astype(np.float32)
    tgt_y = src_y.copy()
    return FeatureBundle(
        cell_id="test__t3a",
        dataset="test",
        model="synth",
        seed=0,
        subject="sub-x",
        source_session="ses-01",
        target_session="ses-02",
        source_features=src_z,
        source_labels=src_y,
        target_features=tgt_z,
        target_y_true=tgt_y,
        n_source=n,
        n_target=n,
        feature_source="synth",
    )


def test_t3a_minimal_runs_without_labels():
    b = _synth_bundle()
    method = MinimalT3AMethod(
        geometry="cosine", filter_k=5, initialization="src_proto", seed=0
    )
    res = run_label_free(method, b)
    assert res.pred.shape == (b.n_target,)
    assert res.used_target_labels is False
    assert set(np.unique(res.pred)).issubset({0, 1})


def test_t3a_rejects_explicit_y_true_kwarg():
    b = _synth_bundle()
    method = MinimalT3AMethod(filter_k=5)
    with pytest.raises(LabelLeakageError):
        run_label_free(method, b, target_y_true=b.target_y_true)


def test_t3a_empty_class_fallback():
    # source missing class 1
    d = 4
    src_z = np.ones((10, d), dtype=np.float32)
    src_y = np.zeros(10, dtype=np.int64)
    tgt_z = np.random.RandomState(1).randn(12, d).astype(np.float32)
    b = FeatureBundle(
        cell_id="empty",
        dataset="t",
        model="m",
        seed=0,
        subject="s",
        source_session="ses-01",
        target_session="ses-02",
        source_features=src_z,
        source_labels=src_y,
        target_features=tgt_z,
        n_source=10,
        n_target=12,
        feature_source="synth",
    )
    res = run_label_free(MinimalT3AMethod(filter_k=3, n_classes=2), b)
    assert res.pred.shape == (12,)
