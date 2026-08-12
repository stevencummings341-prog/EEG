"""Behaviour tests for the cross-subject end-to-end protocol (synthetic data, CPU).

Covers the guarantees that matter for the new mainline:

  * subject-level splits are disjoint (train / val / test), for LOSO, subject k-fold
    and holdout, and are deterministic in ``split_seed``;
  * a cell writes exactly two checkpoints (``best.pt`` + ``last.pt``) and nothing else;
  * re-running a finished cell is a no-op that reuses the stored rows (resume);
  * a cell interrupted after N epochs continues from epoch N+1 instead of restarting;
  * test-subject trials never enter the training or validation loaders;
  * result rows carry both the best-checkpoint and the last-epoch metrics.

Uses 8-channel / 64-sample synthetic sessions and a 1-layer d_model=8 model so the
whole file runs in seconds on CPU.

Run: python -m pytest tests/foundation/test_cross_subject_protocol.py -q
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.datasets.session_splits import load_ok_sessions  # noqa: E402
from code.experiments.cross_subject_protocols import (  # noqa: E402
    _make_dataset as make_trial_dataset,
    load_subject_data,
    make_subject_folds,
    run_cell,
    run_cross_subject,
)
from code.training.e2e_trainer import BEST_CKPT, LAST_CKPT, E2ESpec  # noqa: E402
from code.utils.io import save_session_npz  # noqa: E402

N_CH, N_T = 8, 64
DIMS = dict(n_channels=N_CH, n_times=N_T, n_classes=2, sfreq=250)
TINY_PARAMS = {"d_model": 8, "n_layers": 1, "d_ff": 16, "dino_out_dim": 8, "proto_k": 2}


def _make_dataset(root: Path, n_subjects: int = 6, n_sessions: int = 2,
                  n_trials: int = 16, seed: int = 0) -> Path:
    """Write synthetic per-session npz files + a processed_manifest.csv; return the manifest."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(1, n_subjects + 1):
        subj = f"sub-{s:03d}"
        for ses in range(1, n_sessions + 1):
            sess = f"ses-{ses:02d}"
            y = np.array([0, 1] * (n_trials // 2), dtype=np.int64)
            X = rng.normal(0, 10, size=(n_trials, N_CH, N_T)).astype(np.float32)
            # Weak, subject-shifted class signal so training is not pure noise.
            X[y == 1, :2, :] += 3.0 + 0.2 * s
            path = save_session_npz(
                root / subj / sess / f"{subj}_{sess}.npz", X, y,
                subject_id=subj, session_id=sess, sfreq=250,
                channel_names=[f"C{i}" for i in range(N_CH)],
            )
            rows.append({"subject_id": subj, "session_id": sess, "npz_path": str(path),
                         "status": "ok", "n_trials": n_trials,
                         "label_0_count": int((y == 0).sum()),
                         "label_1_count": int((y == 1).sum())})
    manifest = root / "processed_manifest.csv"
    with open(manifest, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return manifest


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_mi")
    manifest = _make_dataset(root)
    records = load_ok_sessions(manifest)
    subject_data = load_subject_data(records, n_channels=N_CH, n_times=N_T)
    return {"manifest": manifest, "records": records, "subject_data": subject_data}


def _spec(max_epochs: int = 2, **kw) -> E2ESpec:
    base = dict(batch_size=8, lr=1e-3, max_epochs=max_epochs, early_stopping_patience=0,
                num_workers=0, monitor="macro_f1", monitor_mode="max", scheduler="none")
    base.update(kw)
    return E2ESpec(**base)


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def test_pooled_subject_data_has_all_sessions(dataset):
    sd = dataset["subject_data"]
    assert len(sd) == 6
    for subj, data in sd.items():
        assert data.n_trials == 32          # 2 sessions x 16 trials
        assert data.X.shape == (32, N_CH, N_T)
        assert sorted(set(data.sessions)) == ["ses-01", "ses-02"]
        # per-trial z-score was applied at load time
        assert torch.allclose(data.X.mean(dim=-1), torch.zeros(32, N_CH), atol=1e-5)


@pytest.mark.parametrize("protocol,n_folds,expected_folds", [
    ("loso", 0, 6),
    ("kfold_subject", 3, 3),
    ("holdout", 0, 1),
])
def test_folds_are_disjoint_and_cover_subjects(dataset, protocol, n_folds, expected_folds):
    subjects = sorted(dataset["subject_data"])
    folds = make_subject_folds(subjects, protocol=protocol, n_folds=n_folds or 3,
                              split_seed=0, val_subject_fraction=0.25)
    assert len(folds) == expected_folds
    for f in folds:
        f.assert_disjoint()                            # raises if any overlap
        assert set(f.train_subjects + f.val_subjects + f.test_subjects) <= set(subjects)
        assert f.val_subjects, "val subjects should be carved from the train pool"
        assert f.train_sessions is None and f.val_sessions is None
    if protocol != "holdout":
        # every subject is tested exactly once
        tested = [s for f in folds for s in f.test_subjects]
        assert sorted(tested) == subjects


def test_session_val_mode_matches_paper_split(tmp_path):
    """Paper protocol: train ses-01+02, val ses-03, test = all sessions of held-out subject."""
    manifest = _make_dataset(tmp_path, n_subjects=4, n_sessions=3, n_trials=12)
    records = load_ok_sessions(manifest)
    sd = load_subject_data(records, n_channels=N_CH, n_times=N_T)
    folds = make_subject_folds(
        sorted(sd), protocol="loso", val_mode="sessions",
        train_sessions=["ses-01", "ses-02"], val_sessions=["ses-03"], split_seed=0,
    )
    assert len(folds) == 4
    for f in folds:
        f.assert_disjoint()
        assert f.train_subjects == f.val_subjects
        assert set(f.train_subjects).isdisjoint(f.test_subjects)
        assert f.train_sessions == ["ses-01", "ses-02"]
        assert f.val_sessions == ["ses-03"]
        # 3 non-test subjects × 2 train sessions × 12 trials
        _, ytr = make_trial_dataset(sd, f.train_subjects, sessions=f.train_sessions)
        _, yva = make_trial_dataset(sd, f.val_subjects, sessions=f.val_sessions)
        assert int(ytr.numel()) == 3 * 2 * 12
        assert int(yva.numel()) == 3 * 1 * 12
        assert sd[f.test_subjects[0]].n_trials == 3 * 12


def test_session_val_mode_rejects_overlapping_sessions(dataset):
    with pytest.raises(ValueError, match="disjoint"):
        make_subject_folds(
            sorted(dataset["subject_data"]), protocol="loso", val_mode="sessions",
            train_sessions=["ses-01", "ses-02"], val_sessions=["ses-02"],
        )


def test_folds_are_deterministic_in_split_seed(dataset):
    subjects = sorted(dataset["subject_data"])
    a = make_subject_folds(subjects, protocol="kfold_subject", n_folds=3, split_seed=7)
    b = make_subject_folds(subjects, protocol="kfold_subject", n_folds=3, split_seed=7)
    c = make_subject_folds(subjects, protocol="kfold_subject", n_folds=3, split_seed=8)
    assert [f.test_subjects for f in a] == [f.test_subjects for f in b]
    assert [f.test_subjects for f in a] != [f.test_subjects for f in c]


def test_holdout_honours_explicit_subject_lists(dataset):
    subjects = sorted(dataset["subject_data"])
    folds = make_subject_folds(subjects, protocol="holdout", test_subjects=["sub-001"],
                              val_subjects=["sub-002"])
    assert folds[0].test_subjects == ["sub-001"]
    assert folds[0].val_subjects == ["sub-002"]
    assert "sub-001" not in folds[0].train_subjects
    with pytest.raises(ValueError):
        make_subject_folds(subjects, protocol="holdout", test_subjects=["sub-999"])


# --------------------------------------------------------------------------- #
# One cell: checkpoints, rows, resume
# --------------------------------------------------------------------------- #
def test_cell_writes_only_best_and_last_and_reports_both(dataset, tmp_path):
    sd = dataset["subject_data"]
    fold = make_subject_folds(sorted(sd), protocol="kfold_subject", n_folds=3,
                              split_seed=0, val_subject_fraction=0.25)[0]
    ck = tmp_path / "ckpt"
    rows = run_cell(sd, fold, dataset="synthetic", model_name="s4erp",
                    model_params=TINY_PARAMS, data_dims=DIMS, spec=_spec(2), seed=0,
                    device=torch.device("cpu"), ckpt_dir=ck,
                    result_json=tmp_path / "result.json", protocol="kfold_subject")

    assert sorted(p.name for p in ck.iterdir()) == sorted([BEST_CKPT, LAST_CKPT])
    assert len(rows) == len(fold.test_subjects)
    for r in rows:
        assert r["status"] == "ok"
        assert r["test_subject"] in fold.test_subjects
        assert 0.0 <= r["accuracy"] <= 1.0
        assert 0.0 <= r["last_accuracy"] <= 1.0      # final-epoch metrics reported too
        assert r["n_test"] == sd[r["test_subject"]].n_trials
        assert r["n_params"] > 0

    detail = json.loads((tmp_path / "result.json").read_text())
    assert detail["train_info"]["epochs_run"] == 2
    assert len(detail["history"]) == 2
    assert detail["fold"]["test_subjects"] == fold.test_subjects


def test_completed_cell_is_skipped_on_rerun(dataset, tmp_path):
    sd = dataset["subject_data"]
    fold = make_subject_folds(sorted(sd), protocol="kfold_subject", n_folds=3,
                              split_seed=0, val_subject_fraction=0.25)[0]
    kw = dict(dataset="synthetic", model_name="s4erp", model_params=TINY_PARAMS,
              data_dims=DIMS, seed=0, device=torch.device("cpu"),
              ckpt_dir=tmp_path / "ckpt", result_json=tmp_path / "result.json",
              protocol="kfold_subject")
    rows1 = run_cell(sd, fold, spec=_spec(2), **kw)
    mtime = (tmp_path / "ckpt" / LAST_CKPT).stat().st_mtime_ns

    # A longer budget must NOT retrain a cell that already has its result marker.
    rows2 = run_cell(sd, fold, spec=_spec(50), **kw)
    assert rows2 == rows1
    assert (tmp_path / "ckpt" / LAST_CKPT).stat().st_mtime_ns == mtime


def test_interrupted_cell_resumes_from_last_epoch(dataset, tmp_path):
    sd = dataset["subject_data"]
    fold = make_subject_folds(sorted(sd), protocol="kfold_subject", n_folds=3,
                              split_seed=0, val_subject_fraction=0.25)[0]
    ck = tmp_path / "ckpt"
    kw = dict(dataset="synthetic", model_name="s4erp", model_params=TINY_PARAMS,
              data_dims=DIMS, seed=0, device=torch.device("cpu"), ckpt_dir=ck,
              protocol="kfold_subject")

    run_cell(sd, fold, spec=_spec(2), result_json=tmp_path / "r_partial.json", **kw)
    assert torch.load(ck / LAST_CKPT, weights_only=False)["epoch"] == 2

    # Simulate "the job died, resubmit with the same config": the result marker of the
    # full run does not exist yet, so the cell continues from epoch 3.
    run_cell(sd, fold, spec=_spec(4), result_json=tmp_path / "r_full.json", **kw)
    detail = json.loads((tmp_path / "r_full.json").read_text())
    assert detail["train_info"]["resumed_from_epoch"] == 2
    assert detail["train_info"]["epochs_run"] == 4
    assert [h["epoch"] for h in detail["history"]] == [1, 2, 3, 4]
    assert torch.load(ck / LAST_CKPT, weights_only=False)["epoch"] == 4


def test_resume_refuses_a_checkpoint_from_a_different_split(dataset, tmp_path):
    """The dangerous case: same output dir reused after --folds / params changed."""
    sd = dataset["subject_data"]
    folds_a = make_subject_folds(sorted(sd), protocol="kfold_subject", n_folds=3,
                                 split_seed=0, val_subject_fraction=0.25)
    folds_b = make_subject_folds(sorted(sd), protocol="kfold_subject", n_folds=3,
                                 split_seed=99, val_subject_fraction=0.25)
    assert folds_a[0].test_subjects != folds_b[0].test_subjects

    ck = tmp_path / "ckpt"
    kw = dict(dataset="synthetic", model_name="s4erp", model_params=TINY_PARAMS,
              data_dims=DIMS, seed=0, device=torch.device("cpu"), ckpt_dir=ck,
              protocol="kfold_subject")
    run_cell(sd, folds_a[0], spec=_spec(1), result_json=tmp_path / "a.json", **kw)

    with pytest.raises(RuntimeError, match="different configuration"):
        run_cell(sd, folds_b[0], spec=_spec(2), result_json=tmp_path / "b.json", **kw)

    # A completed result marker is likewise protected against config drift.
    with pytest.raises(RuntimeError, match="different configuration"):
        run_cell(sd, folds_a[0], spec=_spec(1), result_json=tmp_path / "a.json",
                 **{**kw, "model_params": {**TINY_PARAMS, "d_model": 16}})

    # --no-resume is the documented escape hatch.
    run_cell(sd, folds_b[0], spec=_spec(1), result_json=tmp_path / "b.json",
             resume=False, **kw)


def test_test_subject_trials_never_reach_training(dataset, tmp_path, monkeypatch):
    """Hard leakage guard: record every trial the trainer is actually handed."""
    from code.experiments import cross_subject_protocols as csp

    sd = dataset["subject_data"]
    fold = make_subject_folds(sorted(sd), protocol="kfold_subject", n_folds=3,
                              split_seed=0, val_subject_fraction=0.25)[0]
    seen: list[torch.Tensor] = []
    real_loader = csp._loader

    def spy(ds, **kw):
        dl = real_loader(ds, **kw)
        # Iterate a second time to record exactly what the loader yields.
        seen.append(torch.cat([xb for xb, _ in dl], dim=0))
        return dl

    monkeypatch.setattr(csp, "_loader", spy)
    run_cell(sd, fold, dataset="synthetic", model_name="s4erp", model_params=TINY_PARAMS,
             data_dims=DIMS, spec=_spec(1), seed=0, device=torch.device("cpu"),
             ckpt_dir=tmp_path / "ckpt", result_json=tmp_path / "result.json",
             protocol="kfold_subject")

    # loaders are created in order: train, val, then one per test subject
    train_val = torch.cat(seen[:2], dim=0)
    n_expected = sum(sd[s].n_trials for s in fold.train_subjects + fold.val_subjects)
    assert len(train_val) == n_expected
    test_trials = torch.cat([sd[s].X for s in fold.test_subjects], dim=0)
    train_val_flat = {t.numpy().tobytes() for t in train_val}
    assert not any(t.numpy().tobytes() in train_val_flat for t in test_trials), \
        "a test-subject trial appeared in the train/val loaders"


# --------------------------------------------------------------------------- #
# Full sweep
# --------------------------------------------------------------------------- #
def test_sweep_covers_models_folds_seeds_and_is_resumable(dataset, tmp_path):
    out = tmp_path / "run"
    kw = dict(dataset="synthetic", models=["s4erp", "dualcd_s4_pos"],
              model_params_all={"s4erp": TINY_PARAMS, "dualcd_s4_pos": TINY_PARAMS},
              data_dims=DIMS, spec=_spec(1), seeds=[0], device=torch.device("cpu"),
              protocol="kfold_subject", n_folds=3, folds_subset=[0],
              val_subject_fraction=0.25, out_dir=out)
    res = run_cross_subject(dataset["records"], **kw)

    assert res["n_cells_failed"] == 0
    assert res["n_subjects"] == 6
    assert {r["model"] for r in res["rows"]} == {"s4erp", "dualcd_s4_pos"}
    assert (out / "splits").exists()
    cells = sorted(p.name for p in (out / "cells").iterdir())
    assert cells == ["dualcd_s4_pos__fold0__seed0", "s4erp__fold0__seed0"]
    for cell in cells:
        assert (out / "cells" / cell / "result.json").exists()
        assert sorted(p.name for p in (out / "cells" / cell / "ckpt").iterdir()) \
            == sorted([BEST_CKPT, LAST_CKPT])

    # second call: everything already complete -> identical rows, no retraining
    res2 = run_cross_subject(dataset["records"], **kw)
    assert res2["rows"] == res["rows"]


def test_channel_mismatch_fails_loudly(dataset):
    with pytest.raises(ValueError, match="expected"):
        load_subject_data(dataset["records"], n_channels=58, n_times=N_T)
