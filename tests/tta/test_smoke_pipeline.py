"""End-to-end tiny smoke pipeline test (synthetic embeddings)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.experiments.session_tta import run_session_tta


def _write_npz(path: Path, n: int = 30, d: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    src_z = rng.randn(n, d).astype(np.float32)
    src_y = np.array([0] * (n // 2) + [1] * (n - n // 2), dtype=np.int64)
    tgt_z = src_z + 0.05 * rng.randn(n, d).astype(np.float32)
    tgt_y = src_y.copy()
    logits = np.zeros((n, 2), dtype=np.float32)
    logits[np.arange(n), src_y] = 2.0
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    pred = probs.argmax(axis=1).astype(np.int64)
    np.savez_compressed(
        path,
        source_train__z=src_z,
        source_train__y=src_y,
        target_test__z=tgt_z,
        target_test__y=tgt_y,
        target_test__logits=logits,
        target_test__probs=probs.astype(np.float32),
        target_test__pred=pred,
        target_test__conf=probs.max(axis=1).astype(np.float32),
    )


def test_smoke_pipeline_synthetic(tmp_path: Path):
    emb = tmp_path / "embeddings"
    # two subjects matching auto-select levels
    for subj in ("sub-stable", "sub-high"):
        _write_npz(emb / "eegnet" / "seed0" / f"{subj}_ses-01-to-ses-02.npz")

    drift = tmp_path / "drift.csv"
    pd.DataFrame(
        [
            {"subject": "sub-high", "drift_level": "high", "drift_score": 1.0},
            {"subject": "sub-stable", "drift_level": "stable", "drift_score": -1.0},
        ]
    ).to_csv(drift, index=False)

    metrics = tmp_path / "metrics.csv"
    rows = []
    for subj in ("sub-stable", "sub-high"):
        rows.append(
            {
                "model": "eegnet",
                "seed": 0,
                "subject": subj,
                "source_session": "ses-01",
                "target_session": "ses-02",
                "prototype_type": "label_based",
                "distance": "euclidean",
                "acc_target": 1.0,
            }
        )
    pd.DataFrame(rows).to_csv(metrics, index=False)

    readable = tmp_path / "readable"
    run_dir = tmp_path / "run"
    cfg = {
        "dataset": "wbci_shu",
        "data": {"n_classes": 2},
        "source_embeddings": {"embeddings_dir": str(emb)},
        "output": {
            "readable_dir": str(readable),
            "run_dir": str(run_dir),
            "output_dir": str(run_dir),
        },
        "round1": {
            "mode": "smoke",
            "max_cells": 2,
            "run_replay_validation": True,
            "run_oracle": True,
            "smoke": {
                "model": "eegnet",
                "model_adapter": "eegnet",
                "seed": 0,
                "drift_summary": str(drift),
                "phase2c_metrics": str(metrics),
                "t3a": {
                    "initialization": "src_proto",
                    "geometry": "cosine",
                    "filter_k": 5,
                    "episodic": False,
                },
            },
        },
    }
    summary = run_session_tta(cfg, project_root=tmp_path)
    assert summary["status"] == "ok"
    assert summary["n_result_rows"] >= 4  # no_tta + t3a (+ oracle) * cells
    assert (readable / "smoke" / "smoke_results.csv").is_file()
    assert (readable / "smoke" / "SMOKE_REPORT.md").is_file()
    assert (readable / "replay_validation" / "REPLAY_VALIDATION_REPORT.md").is_file()
    assert (readable / "oracle_diagnostic" / "ORACLE_DIAGNOSTIC_REPORT.md").is_file()
