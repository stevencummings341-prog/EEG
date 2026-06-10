"""Multi-source cross-session protocol: train ses-01 + ses-02 -> test ses-03.

This Step-1 protocol is kept separate from ``session_protocols.py`` so the
existing within-session and single-source cross-session baselines are untouched.
It reuses the same model builder, trainer, metric set, loaders, and label
normalization as the original baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..datasets.session_splits import SessionRecord, group_by_subject, save_split_json
from ..models.registry import build_model
from ..training.trainer import predict, train_model
from ..utils.seed import set_seed
from .session_protocols import (
    TrainSpec,
    _loader,
    _stratified_val,
    evaluate_predictions,
    load_session_tensors,
)

DEFAULT_TRAIN_SESSIONS: Tuple[str, ...] = ("ses-01", "ses-02")
DEFAULT_TEST_SESSION = "ses-03"
PROTOCOL_NAME = "multisource_0102_to_03"

MULTISOURCE_RESULT_COLUMNS: List[str] = [
    "protocol", "model", "seed", "subject", "train_sessions", "test_session",
    "split_id", "acc", "bacc", "f1", "auc", "nll", "brier", "ece",
    "n_train", "n_val", "n_test", "best_epoch", "checkpoint_path", "status",
    "error_message",
]


def eligible_and_skipped(
    records: Sequence[SessionRecord],
    *,
    train_sessions: Sequence[str] = DEFAULT_TRAIN_SESSIONS,
    test_session: str = DEFAULT_TEST_SESSION,
) -> Tuple[Dict[str, Dict[str, SessionRecord]], List[Dict[str, str]]]:
    """Return eligible subjects and skipped subjects with explicit reasons."""
    required = list(train_sessions) + [test_session]
    used: Dict[str, Dict[str, SessionRecord]] = {}
    skipped: List[Dict[str, str]] = []
    for subj, recs in group_by_subject(records).items():
        sess_map = {r.session: r for r in recs}
        missing = [s for s in required if s not in sess_map]
        if missing:
            skipped.append({
                "subject": subj,
                "reason": f"missing ok session(s): {','.join(missing)}",
                "ok_sessions": "|".join(sorted(sess_map)),
            })
        else:
            used[subj] = sess_map
    return used, skipped


def _fail_row(model: str, seed: int, subject: str, train_label: str,
              test_session: str, message: str) -> Dict[str, object]:
    return {
        "protocol": PROTOCOL_NAME, "model": model, "seed": int(seed),
        "subject": subject, "train_sessions": train_label,
        "test_session": test_session, "split_id": 0,
        "acc": float("nan"), "bacc": float("nan"), "f1": float("nan"),
        "auc": float("nan"), "nll": float("nan"), "brier": float("nan"),
        "ece": float("nan"), "n_train": 0, "n_val": 0, "n_test": 0,
        "best_epoch": -1, "checkpoint_path": "", "status": "failed",
        "error_message": message,
    }


def run_multisource_cross_session(
    records: Sequence[SessionRecord],
    *,
    model_name: str,
    model_params: Dict,
    data_dims: Dict,
    spec: TrainSpec,
    seeds: Sequence[int],
    device: str | torch.device,
    train_sessions: Sequence[str] = DEFAULT_TRAIN_SESSIONS,
    test_session: str = DEFAULT_TEST_SESSION,
    splits_dir: Optional[Path] = None,
    ckpt_dir: Optional[Path] = None,
    logger=None,
) -> Tuple[List[Dict[str, object]], List[str], List[Dict[str, str]]]:
    """Run train=(ses-01+ses-02) -> test=ses-03 for one model."""
    train_sessions = list(train_sessions)
    if test_session in train_sessions:
        raise ValueError("test_session cannot also be a training session")

    dev = torch.device(device)
    pin = dev.type == "cuda"
    train_label = "+".join(train_sessions)
    train_tag = "".join(s.replace("-", "") for s in train_sessions)
    used_map, skipped = eligible_and_skipped(
        records, train_sessions=train_sessions, test_session=test_session)
    used_subjects = sorted(used_map)
    rows: List[Dict[str, object]] = []

    if logger is not None:
        logger.info("[multisource] model=%s | eligible=%d skipped=%d | %s -> %s",
                    model_name, len(used_subjects), len(skipped), train_label, test_session)
        for item in skipped:
            logger.info("[multisource][skip] %s (%s)", item["subject"], item["reason"])

    for subj in used_subjects:
        sess_map = used_map[subj]
        try:
            xs, ys, counts = [], [], {}
            for sess in train_sessions:
                x_s, y_s = load_session_tensors(sess_map[sess].npz_path)
                xs.append(x_s)
                ys.append(y_s)
                counts[sess] = int(len(y_s))
            x_train = torch.cat(xs, dim=0)
            y_train = torch.cat(ys, dim=0)
            x_test, y_test = load_session_tensors(sess_map[test_session].npz_path)
        except Exception as exc:
            for seed in seeds:
                rows.append(_fail_row(model_name, int(seed), subj, train_label,
                                      test_session, f"load_error: {exc}"))
            continue

        all_idx = np.arange(len(y_train))
        y_np = y_train.numpy()
        for seed in seeds:
            try:
                set_seed(int(seed))
                tr_idx, val_idx = _stratified_val(all_idx, y_np, spec.val_fraction, int(seed))
                if val_idx is not None:
                    assert len(np.intersect1d(tr_idx, val_idx)) == 0
                    assert max(np.concatenate([tr_idx, val_idx])) < len(y_train)

                model = build_model(model_name, params=dict(model_params), **data_dims)
                conf_w = float(getattr(model, "confidence_weight", 0.0))
                train_loader = _loader(x_train, y_train, tr_idx, batch_size=spec.batch_size,
                                       shuffle=True, num_workers=spec.num_workers, pin=pin)
                val_loader = None if val_idx is None else _loader(
                    x_train, y_train, val_idx, batch_size=spec.batch_size, shuffle=False,
                    num_workers=spec.num_workers, pin=pin)
                test_idx = np.arange(len(y_test))
                test_loader = _loader(x_test, y_test, test_idx, batch_size=spec.batch_size,
                                      shuffle=False, num_workers=spec.num_workers, pin=pin)

                tinfo = train_model(
                    model, train_loader, val_loader, max_epochs=spec.max_epochs,
                    lr=spec.lr, weight_decay=spec.weight_decay, optimizer=spec.optimizer,
                    early_stopping_patience=spec.early_stopping_patience, device=dev,
                    confidence_weight=conf_w, logger=logger)
                y_true, y_pred, probs, _ = predict(model, test_loader, device=dev)
                metrics = evaluate_predictions(y_true, y_pred, probs)

                cp = ""
                if ckpt_dir is not None:
                    ckpt = (Path(ckpt_dir) / model_name /
                            f"multisource_{subj}_{train_tag}-to-{test_session}_seed{seed}.pt")
                    ckpt.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, ckpt)
                    cp = str(ckpt)

                if splits_dir is not None:
                    save_split_json({
                        "protocol": PROTOCOL_NAME,
                        "subject": subj,
                        "train_sessions": train_sessions,
                        "test_session": test_session,
                        "per_session_train_counts": counts,
                        "n_train": int(len(tr_idx)),
                        "n_val": 0 if val_idx is None else int(len(val_idx)),
                        "n_test": int(len(test_idx)),
                        "seed": int(seed),
                        "train_idx": tr_idx.tolist(),
                        "val_idx": [] if val_idx is None else val_idx.tolist(),
                    }, Path(splits_dir) / f"multisource_{subj}_seed{seed}.json")

                rows.append({
                    "protocol": PROTOCOL_NAME, "model": model_name, "seed": int(seed),
                    "subject": subj, "train_sessions": train_label,
                    "test_session": test_session, "split_id": 0,
                    "acc": float(metrics["accuracy"]),
                    "bacc": float(metrics["balanced_accuracy"]),
                    "f1": float(metrics["macro_f1"]),
                    "auc": float(metrics["auc"]),
                    "nll": float(metrics["nll"]),
                    "brier": float(metrics["brier"]),
                    "ece": float(metrics["ece"]),
                    "n_train": int(len(tr_idx)),
                    "n_val": 0 if val_idx is None else int(len(val_idx)),
                    "n_test": int(len(test_idx)),
                    "best_epoch": int(tinfo["best_epoch"]),
                    "checkpoint_path": cp,
                    "status": "ok",
                    "error_message": "",
                })
                if logger is not None:
                    logger.info("[multisource] %-11s %s seed=%d acc=%.4f (n_train=%d n_val=%d n_test=%d)",
                                model_name, subj, int(seed), float(metrics["accuracy"]),
                                int(len(tr_idx)), 0 if val_idx is None else int(len(val_idx)),
                                int(len(test_idx)))
            except Exception as exc:
                rows.append(_fail_row(model_name, int(seed), subj, train_label,
                                      test_session, f"train_error: {exc}"))
                if logger is not None:
                    logger.error("[multisource] %s seed=%d failed: %s", subj, int(seed), exc)
    return rows, used_subjects, skipped
