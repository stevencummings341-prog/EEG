"""Interface-level label-free safety tests.

Proves that ``run_label_free`` strips ``target_y_true`` BEFORE calling
``method.run`` — an interface-level guarantee, not merely a convention that
well-behaved methods must remember to enforce via ``self._prepare_label_free``.
"""

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
from code.tta.methods.base import MethodResult, TTAMethod
from code.tta.methods.no_tta import NoTTAMethod
from code.tta.methods.t3a_minimal import MinimalT3AMethod
from code.tta.oracle.label_guard import run_label_free, run_oracle
from code.tta.oracle.target_label_proto import TargetLabelProtoOracle


def _bundle_with_labels() -> FeatureBundle:
    rng = np.random.RandomState(0)
    n, d = 20, 6
    z = rng.randn(n, d).astype(np.float32)
    y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int64)
    return FeatureBundle(
        cell_id="label_free_iface_cell",
        dataset="t",
        model="m",
        seed=0,
        subject="s",
        source_session="ses-01",
        target_session="ses-02",
        source_features=z.copy(),
        source_labels=y.copy(),
        target_features=z,
        target_pred=y.copy(),
        target_y_true=y,
        n_source=n,
        n_target=n,
        feature_source="synth",
    )


class MaliciousLabelReadingMethod(TTAMethod):
    """Deliberately malicious test method.

    Does NOT call ``self._prepare_label_free`` and tries to read
    ``bundle.target_y_true`` directly, to simulate a method author who
    forgot (or refused) to respect the label-free convention.
    """

    name = "malicious_label_reader"
    uses_target_labels = False

    def __init__(self) -> None:
        self.observed_y_true = "UNSET-SENTINEL"

    def run(self, bundle: FeatureBundle, **kwargs) -> MethodResult:
        # NOTE: intentionally skips self._prepare_label_free(bundle).
        self.observed_y_true = bundle.target_y_true
        n = bundle.target_features.shape[0]
        pred = np.zeros(n, dtype=np.int64)
        return MethodResult(pred=pred, method=self.name, used_target_labels=False)


def test_run_label_free_strips_labels_before_malicious_method_runs():
    bundle = _bundle_with_labels()
    assert bundle.target_y_true is not None  # sanity: bundle really carries labels

    method = MaliciousLabelReadingMethod()
    result = run_label_free(method, bundle)

    # The malicious method never called _prepare_label_free itself, yet it
    # still observed None: proof the guard strips labels at the interface
    # boundary (before method.run is invoked), not merely by convention.
    assert method.observed_y_true is None
    assert result.used_target_labels is False


def test_run_label_free_does_not_mutate_callers_original_bundle():
    bundle = _bundle_with_labels()
    original_y = bundle.target_y_true.copy()
    run_label_free(MaliciousLabelReadingMethod(), bundle)
    assert bundle.target_y_true is not None
    np.testing.assert_array_equal(bundle.target_y_true, original_y)


def test_run_label_free_rejects_explicit_target_y_true_kwarg():
    bundle = _bundle_with_labels()
    with pytest.raises(LabelLeakageError):
        run_label_free(NoTTAMethod(), bundle, target_y_true=bundle.target_y_true)


def test_run_label_free_rejects_use_target_labels_kwarg():
    bundle = _bundle_with_labels()
    with pytest.raises(LabelLeakageError):
        run_label_free(MinimalT3AMethod(filter_k=3), bundle, use_target_labels=True)


def test_run_label_free_rejects_methods_declaring_uses_target_labels():
    bundle = _bundle_with_labels()

    class DeclaredOracleLikeMethod(TTAMethod):
        name = "declared_oracle_like"
        uses_target_labels = True

        def run(self, bundle: FeatureBundle, **kwargs) -> MethodResult:
            return MethodResult(pred=np.zeros(bundle.n_target, dtype=np.int64), method=self.name)

    with pytest.raises(LabelLeakageError):
        run_label_free(DeclaredOracleLikeMethod(), bundle)


def test_well_behaved_methods_still_work_after_guard_change():
    bundle = _bundle_with_labels()
    r0 = run_label_free(NoTTAMethod(), bundle)
    assert r0.pred.shape == (bundle.n_target,)
    assert r0.used_target_labels is False

    r1 = run_label_free(MinimalT3AMethod(filter_k=5, n_classes=2, seed=0), bundle)
    assert r1.pred.shape == (bundle.n_target,)
    assert r1.used_target_labels is False


def test_oracle_path_still_works_and_still_requires_labels():
    bundle = _bundle_with_labels()
    res = run_oracle(TargetLabelProtoOracle(geometry="cosine"), bundle)
    assert res.used_target_labels is True
    assert res.oracle_diagnostic_only is True
    assert res.not_deployable is True

    stripped = bundle.freeze_for_label_free()
    with pytest.raises(LabelLeakageError):
        run_oracle(TargetLabelProtoOracle(geometry="cosine"), stripped)
