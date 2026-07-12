"""E2E tests for the live ``ModelInferenceSource`` path (test-only mock model).

Exercises the real pipeline: temporary checkpoint -> mock adapter ->
``ModelInferenceSource`` -> ``FeatureBundle`` -> ``no_tta`` -> ``t3a_minimal``
-> evaluator, and proves the checkpoint round-trip is not bypassed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.tta.eval.metrics import build_result_row
from code.tta.eval.schema import validate_result_row
from code.tta.exceptions import CheckpointLoadError, InputValidationError
from code.tta.feature_sources.base import FeatureBundle
from code.tta.feature_sources.model_inference import ModelInferenceSource
from code.tta.methods.no_tta import NoTTAMethod
from code.tta.methods.t3a_minimal import MinimalT3AMethod
from code.tta.oracle.label_guard import run_label_free

from tests.tta.support.mock_adapter import (
    MockModelAdapter,
    save_checkpoint_for_adapter,
    state_dicts_equal,
)

N_CH, N_T, N_CLS, FEAT_DIM = 6, 32, 2, 10


def _make_raw(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(n, N_CH, N_T).astype(np.float32)


def _new_adapter(seed: int, profile: str = "A") -> MockModelAdapter:
    return MockModelAdapter(
        n_channels=N_CH, n_times=N_T, n_classes=N_CLS, feature_dim=FEAT_DIM,
        profile=profile, seed=seed,
    )


def test_checkpoint_actually_saved_and_loaded(tmp_path: Path):
    """Prove the checkpoint round-trip is real, not bypassed."""
    trained = _new_adapter(seed=1)
    fresh = _new_adapter(seed=2)
    assert not state_dicts_equal(trained, fresh), "different seeds must start with different weights"

    ckpt = tmp_path / "mock_ckpt.pt"
    save_checkpoint_for_adapter(trained, ckpt)
    assert ckpt.is_file()

    fresh.load_checkpoint(str(ckpt))
    assert state_dicts_equal(trained, fresh), "weights must match exactly after load"

    x = _make_raw(4)
    np.testing.assert_allclose(trained.forward_logits(x), fresh.forward_logits(x))


def test_checkpoint_missing_raises_typed_error(tmp_path: Path):
    adapter = _new_adapter(seed=0)
    with pytest.raises(CheckpointLoadError):
        adapter.load_checkpoint(str(tmp_path / "does_not_exist.pt"))


def test_model_inference_source_e2e_full_pipeline(tmp_path: Path):
    trained = _new_adapter(seed=7)
    ckpt = tmp_path / "full.pt"
    save_checkpoint_for_adapter(trained, ckpt)

    # A freshly (differently) seeded adapter that MUST load the checkpoint to
    # ever match `trained`'s predictions.
    adapter = _new_adapter(seed=999)
    assert not state_dicts_equal(trained, adapter)
    source = ModelInferenceSource(adapter, device="cpu", batch_size=8, require_features=True)

    n_src, n_tgt = 24, 16
    x_source = _make_raw(n_src, seed=1)
    y_source = np.array(([0, 1] * (n_src // 2))[:n_src], dtype=np.int64)
    x_target = _make_raw(n_tgt, seed=2)
    y_target = np.array(([0, 1] * (n_tgt // 2))[:n_tgt], dtype=np.int64)

    bundle = source.load_cell(
        dataset="mock_ds",
        model="mock_model",
        seed=0,
        subject="sub-e2e",
        source_session="ses-01",
        target_session="ses-02",
        x_target=x_target,
        y_target=y_target,
        x_source=x_source,
        y_source=y_source,
        checkpoint_path=str(ckpt),
    )

    assert isinstance(bundle, FeatureBundle)
    assert bundle.feature_source == "model_inference"
    assert bundle.n_target == n_tgt
    assert bundle.n_source == n_src
    assert bundle.target_features.shape == (n_tgt, FEAT_DIM)
    assert bundle.source_features.shape == (n_src, FEAT_DIM)
    assert bundle.target_logits.shape == (n_tgt, N_CLS)
    assert bundle.target_probs.shape == (n_tgt, N_CLS)
    assert bundle.target_pred.shape == (n_tgt,)
    assert bundle.metadata["capabilities"]["features"] is True
    assert bundle.metadata["checkpoint_path"] == str(ckpt)

    # Proof the checkpoint was really loaded into the *live* adapter used by
    # the source (not bypassed): its target_logits must match a direct
    # forward pass on the independently-checkpointed `trained` adapter.
    direct_logits = trained.forward_logits(x_target)
    np.testing.assert_allclose(bundle.target_logits, direct_logits, rtol=1e-5, atol=1e-6)

    no_tta = NoTTAMethod()
    r0 = run_label_free(no_tta, bundle)
    row0 = build_result_row(bundle=bundle, result=r0, model_adapter=adapter.name)
    assert validate_result_row(row0) == []

    t3a = MinimalT3AMethod(
        geometry="cosine", filter_k=5, initialization="src_proto", n_classes=N_CLS, seed=0
    )
    r1 = run_label_free(t3a, bundle)
    row1 = build_result_row(
        bundle=bundle, result=r1, model_adapter=adapter.name, no_tta_acc=row0["acc"]
    )
    assert validate_result_row(row1) == []
    assert r1.pred.shape == (n_tgt,)
    assert r1.used_target_labels is False


def test_model_inference_source_accepts_dataloader_batches():
    adapter = _new_adapter(seed=3)
    # batch_size on the source is irrelevant when a pre-batched loader is given.
    source = ModelInferenceSource(adapter, batch_size=999)

    x = _make_raw(20, seed=4)
    loader = torch.utils.data.DataLoader(torch.from_numpy(x), batch_size=5, shuffle=False)

    bundle = source.load_cell(
        dataset="mock_ds",
        model="mock_model",
        seed=0,
        subject="sub-loader",
        source_session="ses-01",
        target_session="ses-02",
        x_target=loader,
    )
    assert bundle.n_target == 20
    assert bundle.target_features.shape == (20, FEAT_DIM)

    direct = adapter.forward_features(x)
    np.testing.assert_allclose(bundle.target_features, direct, rtol=1e-5, atol=1e-6)


def test_model_inference_source_shape_mismatch_raises_input_validation_error():
    adapter = _new_adapter(seed=0)
    source = ModelInferenceSource(adapter)
    bad_x = np.random.randn(4, N_CH + 1, N_T).astype(np.float32)  # wrong channel count
    with pytest.raises(InputValidationError):
        source.load_cell(
            dataset="ds", model="m", seed=0, subject="sub",
            source_session="ses-01", target_session="ses-02", x_target=bad_x,
        )


def test_model_inference_source_y_target_length_mismatch_raises():
    adapter = _new_adapter(seed=0)
    source = ModelInferenceSource(adapter)
    x = _make_raw(6)
    with pytest.raises(InputValidationError):
        source.load_cell(
            dataset="ds", model="m", seed=0, subject="sub",
            source_session="ses-01", target_session="ses-02",
            x_target=x, y_target=np.array([0, 1, 0]),
        )


def test_model_inference_source_cell_id_is_canonical():
    adapter = _new_adapter(seed=0)
    source = ModelInferenceSource(adapter, require_features=False)
    x = _make_raw(3)
    bundle = source.load_cell(
        dataset="wbci_shu", model="mock_model", seed=5, subject="sub-001",
        source_session="ses-01", target_session="ses-02", x_target=x,
    )
    assert bundle.cell_id == "wbci_shu__mock_model__seed5__sub-001__ses-01->ses-02"


def test_profile_c_bundle_makes_t3a_minimal_fail_fast():
    """Logits-only Profile C: no_tta ok; MinimalT3AMethod must fail without features."""
    from code.tta.exceptions import UnsupportedCapabilityError
    from code.tta.methods.no_tta import NoTTAMethod
    from code.tta.methods.t3a_minimal import MinimalT3AMethod
    from code.tta.oracle.label_guard import run_label_free

    ad = MockModelAdapter(profile="C", seed=0)
    source = ModelInferenceSource(ad, require_features=False)
    x = _make_raw(8)
    bundle = source.load_cell(
        dataset="ds", model="mock", seed=0, subject="sub",
        source_session="ses-01", target_session="ses-02", x_target=x,
    )
    assert bundle.target_features is None
    assert bundle.target_logits is not None

    r0 = run_label_free(NoTTAMethod(), bundle)
    assert r0.failure_reason == ""
    assert r0.pred is not None

    # T3A requires features — either capability gate or method must fail clearly.
    with pytest.raises((UnsupportedCapabilityError, ValueError, RuntimeError, TypeError)):
        # Prefer gated source when features required:
        gated = ModelInferenceSource(ad, require_features=True)
        gated.load_cell(
            dataset="ds", model="mock", seed=0, subject="sub",
            source_session="ses-01", target_session="ses-02", x_target=x,
        )

    # Direct method on logits-only bundle must also fail fast (no fake embeddings).
    t3a = MinimalT3AMethod(initialization="zeros", filter_k=5, n_classes=2, seed=0)
    with pytest.raises((ValueError, RuntimeError, TypeError, AttributeError)):
        run_label_free(t3a, bundle)
