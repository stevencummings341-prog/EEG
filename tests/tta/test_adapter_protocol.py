"""tests for TTA adapter protocol."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.tta.adapters.embedding_only import EmbeddingOnlyAdapter
from code.tta.adapters.registry import build_adapter, list_adapters
from code.tta.exceptions import UnsupportedAdapterFeature


def test_embedding_only_rejects_forward():
    ad = EmbeddingOnlyAdapter(feature_dim=8, n_classes=2)
    with pytest.raises(UnsupportedAdapterFeature):
        ad.forward_features(np.zeros((1, 58, 1000), dtype=np.float32))
    with pytest.raises(UnsupportedAdapterFeature):
        ad.forward_logits(np.zeros((1, 58, 1000), dtype=np.float32))
    with pytest.raises(UnsupportedAdapterFeature):
        ad.get_classifier_weights()
    assert ad.get_feature_dim() == 8
    meta = ad.get_model_metadata()
    assert meta["adapter_name"] == "embedding_only"


def test_registry_lists_builtins():
    names = list_adapters()
    assert "embedding_only" in names
    assert "eegnet" in names
    ad = build_adapter("embedding_only", feature_dim=4)
    assert ad.name == "embedding_only"


def test_baseline_torch_optional_interface():
    pytest.importorskip("torch")
    ad = build_adapter(
        "eegnet",
        n_channels=58,
        n_times=1000,
        n_classes=2,
        device="cpu",
    )
    x = np.random.randn(2, 58, 1000).astype(np.float32)
    feats = ad.forward_features(x)
    logits = ad.forward_logits(x)
    probs = ad.predict_proba(x)
    assert feats.shape[0] == 2
    assert logits.shape == (2, 2)
    assert probs.shape == (2, 2)
    w = ad.get_classifier_weights()
    assert w.ndim == 2
    assert ad.get_feature_dim() > 0
