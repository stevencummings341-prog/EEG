"""tests for embedding replay feature source."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.tta.exceptions import FeatureSourceError
from code.tta.feature_sources.embedding_replay import (
    EmbeddingReplaySource,
    resolve_embedding_npz_path,
)
from code.tta.feature_sources.base import FeatureBundle


def _make_fake_npz(path: Path, n: int = 20, d: int = 8, c: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    z_s = rng.randn(n, d).astype(np.float32)
    y_s = rng.randint(0, c, size=n).astype(np.int64)
    z_t = rng.randn(n, d).astype(np.float32)
    y_t = rng.randint(0, c, size=n).astype(np.int64)
    logits = rng.randn(n, c).astype(np.float32)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    pred = probs.argmax(axis=1).astype(np.int64)
    conf = probs.max(axis=1).astype(np.float32)
    np.savez_compressed(
        path,
        source_train__z=z_s,
        source_train__y=y_s,
        target_test__z=z_t,
        target_test__y=y_t,
        target_test__logits=logits,
        target_test__probs=probs,
        target_test__pred=pred,
        target_test__conf=conf,
    )


def test_resolve_path_and_replay(tmp_path: Path):
    emb = tmp_path / "embeddings"
    npz = emb / "eegnet" / "seed0" / "sub-001_ses-01-to-ses-02.npz"
    _make_fake_npz(npz)
    resolved = resolve_embedding_npz_path(
        embeddings_dir=emb,
        model="eegnet",
        seed=0,
        subject="sub-001",
        source_session="ses-01",
        target_session="ses-02",
        original_path="/stale/missing.npz",
    )
    assert resolved == npz.resolve()

    src = EmbeddingReplaySource(embeddings_dir=emb, dataset="wbci_shu")
    bundle = src.load_cell(
        model="eegnet",
        seed=0,
        subject="sub-001",
        source_session="ses-01",
        target_session="ses-02",
        original_npz_path="/stale/missing.npz",
    )
    assert isinstance(bundle, FeatureBundle)
    assert bundle.n_target == 20
    assert bundle.target_features is not None
    assert bundle.target_y_true is not None
    assert "wbci_shu__eegnet__seed0__sub-001__ses-01->ses-02" == bundle.cell_id
    frozen = bundle.freeze_for_label_free()
    assert frozen.target_y_true is None


def test_missing_npz_fail_fast(tmp_path: Path):
    with pytest.raises(FeatureSourceError) as ei:
        resolve_embedding_npz_path(
            embeddings_dir=tmp_path / "embeddings",
            model="eegnet",
            seed=0,
            subject="sub-999",
            source_session="ses-01",
            target_session="ses-02",
            original_path="/does/not/exist.npz",
        )
    msg = str(ei.value)
    assert "sub-999" in msg
    assert "tried_paths" in msg
