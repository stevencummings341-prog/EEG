"""Tests for explicit adapter capability reporting and fail-fast behavior."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.tta.adapters.base import (
    AdapterCapabilities,
    ModelAdapter,
    require_capability,
    soft_call,
)
from code.tta.adapters.embedding_only import EmbeddingOnlyAdapter
from code.tta.exceptions import UnsupportedAdapterFeature, UnsupportedCapabilityError
from code.tta.feature_sources.model_inference import ModelInferenceSource

from tests.tta.support.mock_adapter import MockModelAdapter


def test_unsupported_capability_error_is_a_subclass_for_compat():
    assert issubclass(UnsupportedCapabilityError, UnsupportedAdapterFeature)


def test_default_capabilities_are_conservative():
    class BareAdapter(ModelAdapter):
        name = "bare"

    caps = BareAdapter().capabilities()
    assert isinstance(caps, AdapterCapabilities)
    assert caps.features is False
    assert caps.logits is False
    assert caps.probabilities is False
    assert caps.classifier_weights is False
    assert caps.source_prototypes is False
    assert caps.checkpoint_loading is False
    assert caps.input_validation is False
    assert caps.metadata is True  # every adapter implements get_model_metadata


def test_embedding_only_adapter_declares_no_live_capabilities():
    ad = EmbeddingOnlyAdapter(feature_dim=8, n_classes=2)
    caps = ad.capabilities()
    assert caps.features is False
    assert caps.logits is False
    assert caps.probabilities is False
    assert caps.checkpoint_loading is False
    assert caps.metadata is True


def test_baseline_torch_adapter_reports_full_capabilities():
    pytest.importorskip("torch")
    from code.tta.adapters.baseline_torch import BaselineTorchAdapter

    ad = BaselineTorchAdapter(model_name="eegnet", n_channels=8, n_times=64, n_classes=2)
    caps = ad.capabilities()
    assert caps.features is True
    assert caps.logits is True
    assert caps.probabilities is True
    assert caps.checkpoint_loading is True
    assert caps.input_validation is True
    assert isinstance(caps.classifier_weights, bool)


def test_profile_a_full_capabilities():
    ad = MockModelAdapter(profile="A")
    caps = ad.capabilities()
    assert caps.features and caps.logits and caps.probabilities
    assert caps.classifier_weights is True
    assert caps.checkpoint_loading is True
    w = ad.get_classifier_weights()
    assert w.ndim == 2


def test_profile_b_missing_classifier_weights_fails_fast():
    ad = MockModelAdapter(profile="B")
    caps = ad.capabilities()
    assert caps.features and caps.logits
    assert caps.classifier_weights is False
    with pytest.raises(UnsupportedCapabilityError):
        ad.get_classifier_weights()
    # subclass compat: legacy handlers catching the old exception still work.
    with pytest.raises(UnsupportedAdapterFeature):
        ad.get_classifier_weights()


def test_profile_c_logits_only_no_fake_features():
    ad = MockModelAdapter(profile="C")
    caps = ad.capabilities()
    assert caps.features is False
    assert caps.logits is True
    assert caps.probabilities is True
    x = np.random.randn(2, ad.n_channels, ad.n_times).astype(np.float32)
    with pytest.raises(UnsupportedCapabilityError):
        ad.forward_features(x)
    assert ad.forward_logits(x).shape == (2, ad.n_classes)
    assert ad.predict_proba(x).shape == (2, ad.n_classes)


def test_require_capability_fails_fast_with_typed_error():
    ad = MockModelAdapter(profile="C")
    with pytest.raises(UnsupportedCapabilityError):
        require_capability(ad, "features", context="t3a needs embeddings")
    # Present capability: no error, returns the AdapterCapabilities.
    caps = require_capability(ad, "logits")
    assert caps.logits is True


def test_soft_call_is_documented_test_only_silent_default():
    ad = MockModelAdapter(profile="C")
    x = np.zeros((1, ad.n_channels, ad.n_times), dtype=np.float32)
    result = soft_call(ad, "forward_features", x, default="MISSING")
    assert result == "MISSING"
    assert "test-only" in soft_call.__doc__.lower() or "test only" in soft_call.__doc__.lower()


def test_model_inference_source_fails_fast_when_features_required_but_unsupported():
    ad = MockModelAdapter(profile="C")
    source = ModelInferenceSource(ad, require_features=True)  # default
    x = np.random.randn(4, ad.n_channels, ad.n_times).astype(np.float32)
    with pytest.raises(UnsupportedCapabilityError):
        source.load_cell(
            dataset="ds", model="m", seed=0, subject="sub",
            source_session="ses-01", target_session="ses-02", x_target=x,
        )


def test_model_inference_source_profile_c_ok_for_no_tta_when_features_not_required():
    ad = MockModelAdapter(profile="C")
    source = ModelInferenceSource(ad, require_features=False)
    x = np.random.randn(4, ad.n_channels, ad.n_times).astype(np.float32)
    bundle = source.load_cell(
        dataset="ds", model="m", seed=0, subject="sub",
        source_session="ses-01", target_session="ses-02", x_target=x,
    )
    assert bundle.target_features is None  # no fake embeddings fabricated
    assert bundle.target_logits is not None
    assert bundle.target_pred is not None


def test_model_inference_source_profile_b_features_and_logits_ok_but_weights_fail_fast():
    ad = MockModelAdapter(profile="B")
    source = ModelInferenceSource(ad, require_features=True, require_classifier_weights=True)
    x = np.random.randn(3, ad.n_channels, ad.n_times).astype(np.float32)
    with pytest.raises(UnsupportedCapabilityError):
        source.load_cell(
            dataset="ds", model="m", seed=0, subject="sub",
            source_session="ses-01", target_session="ses-02", x_target=x,
        )

    source_no_weights = ModelInferenceSource(ad, require_features=True)
    bundle = source_no_weights.load_cell(
        dataset="ds", model="m", seed=0, subject="sub",
        source_session="ses-01", target_session="ses-02", x_target=x,
    )
    assert bundle.target_features is not None
    assert bundle.target_logits is not None


def test_model_inference_source_checkpoint_capability_required():
    class NoCheckpointAdapter(ModelAdapter):
        name = "no_ckpt"

        def capabilities(self) -> AdapterCapabilities:
            return AdapterCapabilities(features=True, logits=True, probabilities=True)

        def forward_features(self, x):
            return np.zeros((len(x), 4), dtype=np.float32)

        def forward_logits(self, x):
            return np.zeros((len(x), 2), dtype=np.float32)

        def predict_proba(self, x):
            return np.full((len(x), 2), 0.5, dtype=np.float32)

    ad = NoCheckpointAdapter()
    source = ModelInferenceSource(ad)
    x = np.random.randn(3, 4, 4).astype(np.float32)
    with pytest.raises(UnsupportedCapabilityError):
        source.load_cell(
            dataset="ds", model="m", seed=0, subject="sub",
            source_session="ses-01", target_session="ses-02", x_target=x,
            checkpoint_path="/tmp/whatever_does_not_matter.pt",
        )
