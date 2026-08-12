"""Within-session CV and cross-session evaluation protocols (fair, leakage-free).

Both protocols use the SAME model builder, trainer, metric set, and data filter so
baselines (EEGNet/DeepConvNet/FBCNet) and CAP-EEGNet are compared apples-to-apples.

  within_session : for each ok session, Stratified K-fold CV over its 200 trials.
                   train/test are disjoint trials of the SAME session (no drift).
                   A small stratified val slice is carved from each fold's train set
                   for early stopping. -> upper bound performance.

  cross_session  : for each subject, every DIRECTED ok session pair
                   (train_session -> test_session). Train on ALL train_session trials
                   (minus a val slice), test on ALL test_session trials. -> drift.

Leakage rules enforced here:
  * test trials never appear in train or val (within: disjoint fold; cross: different
    session entirely).
  * the early-stopping val slice is carved ONLY from train, never from test.
  * labels normalized to {0,1}; stratification keeps the class balance.

Metrics per run: Accuracy, Balanced Accuracy, Macro-F1, AUC, NLL, Brier, ECE.
Each run row + the exact split (JSON) + (optionally) the best-fold checkpoint are
persisted for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..datasets.session_splits import (
    SessionRecord,
    build_cross_split_record,
    build_within_split_record,
    group_by_subject,
    make_cross_session_pairs,
    make_within_session_folds,
    normalize_labels,
    save_split_json,
)
from ..models.registry import build_model
from ..utils.io import load_session_npz
from ..utils.seed import set_seed
from .metrics import (
    auc_binary,
    auc_multiclass,
    brier_score,
    classification_metrics,
    expected_calibration_error,
    negative_log_likelihood,
)
from ..training.trainer import predict, train_model


@dataclass
class TrainSpec:
    """Shared training hyperparameters (from configs/session_model_compare.yaml train)."""

    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"
    max_epochs: int = 100
    early_stopping_patience: int = 20
    val_fraction: float = 0.2
    num_workers: int = 2


# --------------------------------------------------------------------------- #
# Data + metrics helpers
# --------------------------------------------------------------------------- #
def load_session_tensors(npz_path: str | Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load one session .npz -> (X [n,58,1000] float32, y [n] int64 in {0,1})."""
    d = load_session_npz(npz_path)
    X = torch.from_numpy(np.ascontiguousarray(d["X"], dtype=np.float32))
    y = torch.from_numpy(normalize_labels(d["y"]).astype(np.int64))
    return X, y


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    """Accuracy / Balanced Accuracy / Macro-F1 / AUC / NLL / Brier / ECE.

    AUC: binary uses P(class=1); multiclass (C>=3) uses macro-OVR.
    """
    m = classification_metrics(y_true, y_pred)
    probs = np.asarray(probs)
    if probs.ndim != 2:
        raise ValueError(f"probs must be [N,C], got shape {probs.shape}")
    if probs.shape[1] <= 2:
        m["auc"] = auc_binary(y_true, probs[:, min(1, probs.shape[1] - 1)])
    else:
        m["auc"] = auc_multiclass(y_true, probs)
    m["nll"] = negative_log_likelihood(y_true, probs)
    m["brier"] = brier_score(y_true, probs)
    m["ece"] = expected_calibration_error(y_true, probs)
    return m


def _stratified_val(idx: np.ndarray, labels: np.ndarray, val_fraction: float, seed: int
                    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Split ``idx`` into (train, val) stratified by ``labels`` (aligned to idx)."""
    if val_fraction <= 0:
        return idx, None
    from sklearn.model_selection import train_test_split
    # Need at least one sample per class on each side; fall back to no-val if tiny.
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2 or counts.min() < 2:
        return idx, None
    tr, va = train_test_split(idx, test_size=val_fraction, random_state=seed, stratify=labels)
    return tr, va


def _loader(X: torch.Tensor, y: torch.Tensor, idx: np.ndarray, *, batch_size: int,
            shuffle: bool, num_workers: int, pin: bool) -> DataLoader:
    sel = torch.as_tensor(idx, dtype=torch.long)
    ds = TensorDataset(X[sel], y[sel])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin, drop_last=False)


def _device(device: str | torch.device) -> torch.device:
    return torch.device(device)


# --------------------------------------------------------------------------- #
# Within-session CV
# --------------------------------------------------------------------------- #
def run_within_session(
    records: Sequence[SessionRecord],
    *,
    model_name: str,
    model_params: Dict,
    data_dims: Dict,
    spec: TrainSpec,
    folds: int,
    seeds: Sequence[int],
    device: str | torch.device,
    splits_dir: Optional[Path] = None,
    ckpt_dir: Optional[Path] = None,
    logger=None,
) -> List[Dict[str, object]]:
    """Run within-session K-fold CV for one model over all given ok sessions."""
    dev = _device(device)
    pin = dev.type == "cuda"
    rows: List[Dict[str, object]] = []

    for rec in records:
        X, y = load_session_tensors(rec.npz_path)
        y_np = y.numpy()
        for seed in seeds:
            set_seed(int(seed))
            if splits_dir is not None:
                rec_json = build_within_split_record(rec.subject, rec.session, y_np, folds, int(seed))
                save_split_json(rec_json, Path(splits_dir) / f"within_{rec.subject}_{rec.session}_seed{seed}.json")

            fold_splits = make_within_session_folds(rec.subject, rec.session, y_np,
                                                    n_splits=folds, seed=int(seed))
            best_fold_val = float("inf")
            best_fold_state = None
            for fold, (tr_idx, te_idx) in enumerate(fold_splits):
                tr2, va = _stratified_val(tr_idx, y_np[tr_idx], spec.val_fraction, int(seed))
                model = build_model(model_name, params=dict(model_params), **data_dims)
                conf_w = float(getattr(model, "confidence_weight", 0.0))
                train_loader = _loader(X, y, tr2, batch_size=spec.batch_size, shuffle=True,
                                       num_workers=spec.num_workers, pin=pin)
                val_loader = (None if va is None else
                              _loader(X, y, va, batch_size=spec.batch_size, shuffle=False,
                                      num_workers=spec.num_workers, pin=pin))
                test_loader = _loader(X, y, te_idx, batch_size=spec.batch_size, shuffle=False,
                                      num_workers=spec.num_workers, pin=pin)

                tinfo = train_model(
                    model, train_loader, val_loader,
                    max_epochs=spec.max_epochs, lr=spec.lr, weight_decay=spec.weight_decay,
                    optimizer=spec.optimizer, early_stopping_patience=spec.early_stopping_patience,
                    device=dev, confidence_weight=conf_w, logger=logger,
                )
                y_true, y_pred, probs, _ = predict(model, test_loader, device=dev)
                metrics = evaluate_predictions(y_true, y_pred, probs)
                rows.append({
                    "model": model_name, "protocol": "within_session",
                    "subject": rec.subject, "session": rec.session,
                    "train_session": "", "test_session": "",
                    "seed": int(seed), "fold": fold,
                    "n_train": int(len(tr2)), "n_val": (0 if va is None else int(len(va))),
                    "n_test": int(len(te_idx)), "best_epoch": int(tinfo["best_epoch"]),
                    **{k: float(v) for k, v in metrics.items()},
                })
                if tinfo["best_val_loss"] < best_fold_val:
                    best_fold_val = float(tinfo["best_val_loss"])
                    best_fold_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            if ckpt_dir is not None and best_fold_state is not None:
                cp = Path(ckpt_dir) / model_name / f"within_{rec.subject}_{rec.session}_seed{seed}.pt"
                cp.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_fold_state, cp)
            if logger is not None:
                accs = [r["accuracy"] for r in rows
                        if r["subject"] == rec.subject and r["session"] == rec.session and r["seed"] == seed]
                logger.info("[within] %-11s %s/%s seed=%d acc=%.4f (folds=%d)",
                            model_name, rec.subject, rec.session, seed,
                            float(np.mean(accs)), len(accs))
    return rows


# --------------------------------------------------------------------------- #
# Cross-session (directed pairs)
# --------------------------------------------------------------------------- #
def run_cross_session(
    records: Sequence[SessionRecord],
    *,
    model_name: str,
    model_params: Dict,
    data_dims: Dict,
    spec: TrainSpec,
    seeds: Sequence[int],
    device: str | torch.device,
    splits_dir: Optional[Path] = None,
    ckpt_dir: Optional[Path] = None,
    logger=None,
) -> List[Dict[str, object]]:
    """Run cross-session (train ses-i -> test ses-j) for one model over all subjects."""
    dev = _device(device)
    pin = dev.type == "cuda"
    rows: List[Dict[str, object]] = []
    by_subj = group_by_subject(records)

    if splits_dir is not None:
        save_split_json(build_cross_split_record(records), Path(splits_dir) / "cross_session_pairs.json")

    for subj, recs in by_subj.items():
        if len(recs) < 2:
            if logger is not None:
                logger.info("[cross] %s has <2 ok sessions; skipped.", subj)
            continue
        sess_tensors = {r.session: load_session_tensors(r.npz_path) for r in recs}
        pairs = make_cross_session_pairs(subj, [r.session for r in recs])
        for seed in seeds:
            for pair in pairs:
                set_seed(int(seed))
                Xtr, ytr = sess_tensors[pair["train_session"]]
                Xte, yte = sess_tensors[pair["test_session"]]
                ytr_np = ytr.numpy()
                all_idx = np.arange(len(ytr_np))
                tr2, va = _stratified_val(all_idx, ytr_np, spec.val_fraction, int(seed))

                model = build_model(model_name, params=dict(model_params), **data_dims)
                conf_w = float(getattr(model, "confidence_weight", 0.0))
                train_loader = _loader(Xtr, ytr, tr2, batch_size=spec.batch_size, shuffle=True,
                                       num_workers=spec.num_workers, pin=pin)
                val_loader = (None if va is None else
                              _loader(Xtr, ytr, va, batch_size=spec.batch_size, shuffle=False,
                                      num_workers=spec.num_workers, pin=pin))
                test_idx = np.arange(len(yte))
                test_loader = _loader(Xte, yte, test_idx, batch_size=spec.batch_size, shuffle=False,
                                      num_workers=spec.num_workers, pin=pin)

                tinfo = train_model(
                    model, train_loader, val_loader,
                    max_epochs=spec.max_epochs, lr=spec.lr, weight_decay=spec.weight_decay,
                    optimizer=spec.optimizer, early_stopping_patience=spec.early_stopping_patience,
                    device=dev, confidence_weight=conf_w, logger=logger,
                )
                y_true, y_pred, probs, _ = predict(model, test_loader, device=dev)
                metrics = evaluate_predictions(y_true, y_pred, probs)
                rows.append({
                    "model": model_name, "protocol": "cross_session",
                    "subject": subj, "session": "",
                    "train_session": pair["train_session"], "test_session": pair["test_session"],
                    "seed": int(seed), "fold": "",
                    "n_train": int(len(tr2)), "n_val": (0 if va is None else int(len(va))),
                    "n_test": int(len(test_idx)), "best_epoch": int(tinfo["best_epoch"]),
                    **{k: float(v) for k, v in metrics.items()},
                })
                if ckpt_dir is not None:
                    cp = (Path(ckpt_dir) / model_name /
                          f"cross_{subj}_{pair['train_session']}-to-{pair['test_session']}_seed{seed}.pt")
                    cp.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, cp)
            if logger is not None:
                acc = float(np.mean([r["accuracy"] for r in rows if r["subject"] == subj and r["seed"] == seed]))
                logger.info("[cross] %-11s %s seed=%d mean_acc=%.4f (%d pairs)",
                            model_name, subj, seed, acc, len(pairs))
    return rows


# --------------------------------------------------------------------------- #
# CSV column order (shared by trainer outputs + summarizer)
# --------------------------------------------------------------------------- #
RESULT_COLUMNS: List[str] = [
    "model", "protocol", "subject", "session", "train_session", "test_session",
    "seed", "fold", "n_train", "n_val", "n_test", "best_epoch",
    "accuracy", "balanced_accuracy", "macro_f1", "auc", "nll", "brier", "ece",
]
