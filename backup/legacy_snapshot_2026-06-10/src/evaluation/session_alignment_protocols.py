"""Step-2 no-learning / unsupervised test-time alignment protocols.

Covers both cross-session protocol groups, reusing the baseline trainer, model
registry, metrics, loaders, and stratified-val split:

  * single-source directed pairs: train ses-i -> test ses-j (both ok).
  * multi-source: train ses-01+ses-02 -> test ses-03 (all three ok).

For every (task, method, model, seed) we:
  1. concatenate the source-session train trials, carve a stratified val slice
     from the SOURCE TRAIN only (never from the target);
  2. for feature-space methods, fit the source alignment on the SOURCE TRAIN
     slice and apply it to source train + source val; align the target with the
     target X (unlabeled) — except ``filterbank_reweighting`` which reweights the
     target toward the SOURCE band-power profile;
  3. train the model (early stopping on the source val) — for
     ``bn_statistics_adaptation`` no alignment is applied; after training we
     forward the unlabeled target X to refresh BN running stats only;
  4. evaluate on the target test set (``y_test`` used ONLY here).

Leakage guarantees (asserted per run): target trials never enter train/val;
target labels never enter training/validation/early-stopping/selection; no
``optimizer.step`` on the target (only BN running-stat updates for the BN
method). Each row records ``used_target_x_for_stats`` and
``used_target_y_for_training`` (always False).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..adaptation.bn_adaptation import adapt_bn_statistics, count_bn_layers
from ..adaptation.session_alignment import ALIGNMENT_METHODS, make_alignment_method
from ..data.session_splits import (
    SessionRecord,
    group_by_subject,
    make_cross_session_pairs,
)
from ..models.registry import build_model
from ..training.trainer import predict, train_model
from ..utils.seed import set_seed
from .session_protocols import TrainSpec, _loader, _stratified_val, evaluate_predictions, load_session_tensors

EXPERIMENT_ID = "alignment_baseline_v1"
BN_METHOD = "bn_statistics_adaptation"
NONE_METHOD = "none_reference"
ALL_METHODS = [NONE_METHOD] + ALIGNMENT_METHODS[:3] + [BN_METHOD, ALIGNMENT_METHODS[3]]
# Methods actually trained by this protocol (none_reference is pulled from baseline_v1):
TRAINED_METHODS = ALIGNMENT_METHODS[:3] + [BN_METHOD, ALIGNMENT_METHODS[3]]

MULTISOURCE_TRAIN_SESSIONS: Tuple[str, ...] = ("ses-01", "ses-02")
MULTISOURCE_TEST_SESSION = "ses-03"

ALIGNMENT_RESULT_COLUMNS: List[str] = [
    "experiment_id", "method", "protocol", "training_scope", "model", "seed",
    "subject", "train_sessions", "test_session",
    "acc", "bacc", "f1", "auc", "nll", "brier", "ece",
    "n_train", "n_val", "n_test",
    "source_alignment_stats", "target_alignment_stats",
    "used_target_x_for_stats", "used_target_y_for_training",
    "checkpoint_path", "status", "error_message",
]


# --------------------------------------------------------------------------- #
# Task enumeration
# --------------------------------------------------------------------------- #
@dataclass
class AlignTask:
    subject: str
    training_scope: str           # "single_source" | "multi_source"
    protocol: str                 # e.g. "ses-01->ses-03" | "ses-01+ses-02->ses-03"
    train_sessions: List[str]
    test_session: str
    src_npz: Dict[str, str]       # session -> npz path
    test_npz: str
    tag: str = field(default="")  # filename-safe id

    def train_label(self) -> str:
        return "+".join(self.train_sessions)


def _tag(scope: str, subject: str, train_sessions: Sequence[str], test_session: str) -> str:
    tr = "".join(s.replace("-", "") for s in train_sessions)
    te = test_session.replace("-", "")
    return f"{scope}_{subject}_{tr}-to-{te}"


def enumerate_single_source_tasks(records: Sequence[SessionRecord]) -> List[AlignTask]:
    tasks: List[AlignTask] = []
    for subj, recs in group_by_subject(records).items():
        if len(recs) < 2:
            continue
        npz_by_sess = {r.session: r.npz_path for r in recs}
        for pair in make_cross_session_pairs(subj, [r.session for r in recs]):
            tr, te = pair["train_session"], pair["test_session"]
            tasks.append(AlignTask(
                subject=subj, training_scope="single_source",
                protocol=f"{tr}->{te}", train_sessions=[tr], test_session=te,
                src_npz={tr: npz_by_sess[tr]}, test_npz=npz_by_sess[te],
                tag=_tag("single", subj, [tr], te),
            ))
    return tasks


def enumerate_multisource_tasks(
    records: Sequence[SessionRecord],
    *,
    train_sessions: Sequence[str] = MULTISOURCE_TRAIN_SESSIONS,
    test_session: str = MULTISOURCE_TEST_SESSION,
) -> Tuple[List[AlignTask], List[Dict[str, str]]]:
    required = list(train_sessions) + [test_session]
    tasks: List[AlignTask] = []
    skipped: List[Dict[str, str]] = []
    for subj, recs in group_by_subject(records).items():
        npz_by_sess = {r.session: r.npz_path for r in recs}
        missing = [s for s in required if s not in npz_by_sess]
        if missing:
            skipped.append({"subject": subj, "reason": f"missing ok session(s): {','.join(missing)}",
                            "ok_sessions": "|".join(sorted(npz_by_sess))})
            continue
        tasks.append(AlignTask(
            subject=subj, training_scope="multi_source",
            protocol=f"{'+'.join(train_sessions)}->{test_session}",
            train_sessions=list(train_sessions), test_session=test_session,
            src_npz={s: npz_by_sess[s] for s in train_sessions},
            test_npz=npz_by_sess[test_session],
            tag=_tag("multi", subj, train_sessions, test_session),
        ))
    return tasks, skipped


# --------------------------------------------------------------------------- #
# One run (task, method, model, seed)
# --------------------------------------------------------------------------- #
def _fail_row(method: str, task: AlignTask, model_name: str, seed: int, message: str,
              used_target_x: bool) -> Dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID, "method": method, "protocol": task.protocol,
        "training_scope": task.training_scope, "model": model_name, "seed": int(seed),
        "subject": task.subject, "train_sessions": task.train_label(),
        "test_session": task.test_session,
        "acc": float("nan"), "bacc": float("nan"), "f1": float("nan"), "auc": float("nan"),
        "nll": float("nan"), "brier": float("nan"), "ece": float("nan"),
        "n_train": 0, "n_val": 0, "n_test": 0,
        "source_alignment_stats": "{}", "target_alignment_stats": "{}",
        "used_target_x_for_stats": bool(used_target_x), "used_target_y_for_training": False,
        "checkpoint_path": "", "status": "failed", "error_message": message,
    }


def _json(o) -> str:
    return json.dumps(o, ensure_ascii=False, sort_keys=True, default=float)


def _atomic_save_split(obj: Dict[str, object], path: Path) -> None:
    """Concurrency-safe split write: many parallel jobs share the same (task,seed) file.

    Identical content is produced regardless of method/model, so a temp-file +
    os.replace (atomic on POSIX) makes the last writer win without corruption.
    """
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


@dataclass
class _Preloaded:
    """Per-task tensors + per-seed split (shared across methods/models)."""
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    x_train_np: np.ndarray
    x_test_np: np.ndarray
    counts: Dict[str, int]


def _preload(task: AlignTask) -> _Preloaded:
    xs, ys, counts = [], [], {}
    for sess in task.train_sessions:
        x_s, y_s = load_session_tensors(task.src_npz[sess])
        xs.append(x_s); ys.append(y_s); counts[sess] = int(len(y_s))
    x_train = torch.cat(xs, dim=0)
    y_train = torch.cat(ys, dim=0)
    x_test, y_test = load_session_tensors(task.test_npz)
    return _Preloaded(
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        x_train_np=x_train.numpy().astype(np.float64),
        x_test_np=x_test.numpy().astype(np.float64), counts=counts,
    )


def _align_arrays(method: str, pre: _Preloaded, tr_idx: np.ndarray, sfreq: int, params: Dict
                  ) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """Return (X_train_aligned, X_test_aligned, source_stats_json, target_stats_json)."""
    src_tf = make_alignment_method(method, sfreq=sfreq, params=params)
    src_tf.fit(pre.x_train_np[tr_idx])
    x_train_al = src_tf.transform(pre.x_train_np)
    src_summary = src_tf.summary()

    if method == "filterbank_reweighting":
        # Reweight the target toward the SOURCE band-power profile (target X only).
        x_test_al = src_tf.transform(pre.x_test_np)
        tgt_summary = src_tf.summary()
    else:
        tgt_tf = make_alignment_method(method, sfreq=sfreq, params=params)
        tgt_tf.fit(pre.x_test_np)
        x_test_al = tgt_tf.transform(pre.x_test_np)
        tgt_summary = tgt_tf.summary()
    return x_train_al, x_test_al, _json(src_summary), _json(tgt_summary)


def run_alignment_tasks(
    tasks: Sequence[AlignTask],
    *,
    methods: Sequence[str],
    models: Sequence[str],
    model_params_all: Dict[str, Dict],
    data_dims: Dict,
    spec: TrainSpec,
    seeds: Sequence[int],
    device: str | torch.device,
    sfreq: int = 250,
    align_params: Optional[Dict] = None,
    splits_dir: Optional[Path] = None,
    ckpt_dir: Optional[Path] = None,
    save_ckpt: bool = True,
    logger=None,
) -> List[Dict[str, object]]:
    """Run all (task, method, model, seed) combinations; one result row each."""
    dev = torch.device(device)
    pin = dev.type == "cuda"
    align_params = dict(align_params or {})
    methods = [m for m in methods if m != NONE_METHOD]
    rows: List[Dict[str, object]] = []

    for task in tasks:
        try:
            pre = _preload(task)
        except Exception as exc:
            for method in methods:
                for model_name in models:
                    for seed in seeds:
                        rows.append(_fail_row(method, task, model_name, int(seed),
                                              f"load_error: {exc}", used_target_x=method != NONE_METHOD))
            if logger is not None:
                logger.error("[align] %s load failed: %s", task.tag, exc)
            continue

        y_train_np = pre.y_train.numpy()
        all_idx = np.arange(len(y_train_np))
        test_idx = np.arange(len(pre.y_test))

        for seed in seeds:
            set_seed(int(seed))
            tr_idx, val_idx = _stratified_val(all_idx, y_train_np, spec.val_fraction, int(seed))
            # Leakage guards (split level).
            if val_idx is not None:
                assert len(np.intersect1d(tr_idx, val_idx)) == 0
                assert int(np.max(np.concatenate([tr_idx, val_idx]))) < len(y_train_np)

            if splits_dir is not None:
                _atomic_save_split({
                    "experiment_id": EXPERIMENT_ID, "training_scope": task.training_scope,
                    "protocol": task.protocol, "subject": task.subject,
                    "train_sessions": task.train_sessions, "test_session": task.test_session,
                    "per_session_train_counts": pre.counts,
                    "n_train": int(len(tr_idx)),
                    "n_val": 0 if val_idx is None else int(len(val_idx)),
                    "n_test": int(len(test_idx)), "seed": int(seed),
                    "train_idx": tr_idx.tolist(),
                    "val_idx": [] if val_idx is None else val_idx.tolist(),
                }, Path(splits_dir) / f"{task.tag}_seed{seed}.json")

            for method in methods:
                used_target_x = True  # all trained methods use unlabeled target X
                # Build the (possibly aligned) train/test tensors for this method+seed.
                try:
                    if method == BN_METHOD:
                        x_train_t = pre.x_train
                        x_test_t = pre.x_test
                        src_stats = _json({"method": BN_METHOD})
                        tgt_stats = None  # filled after adaptation (needs the model)
                        aligned = False
                    else:
                        x_tr_al, x_te_al, src_stats, tgt_stats = _align_arrays(
                            method, pre, tr_idx, sfreq, align_params)
                        # Shape invariants.
                        assert x_tr_al.shape == pre.x_train_np.shape, "alignment changed train X shape"
                        assert x_te_al.shape == pre.x_test_np.shape, "alignment changed test X shape"
                        assert np.all(np.isfinite(x_tr_al)) and np.all(np.isfinite(x_te_al)), \
                            "alignment produced NaN/Inf"
                        x_train_t = torch.from_numpy(np.ascontiguousarray(x_tr_al, dtype=np.float32))
                        x_test_t = torch.from_numpy(np.ascontiguousarray(x_te_al, dtype=np.float32))
                        aligned = True
                except Exception as exc:
                    for model_name in models:
                        rows.append(_fail_row(method, task, model_name, int(seed),
                                              f"align_error: {exc}", used_target_x))
                    if logger is not None:
                        logger.error("[align] %s %s seed=%d align failed: %s",
                                     task.tag, method, int(seed), exc)
                    continue

                for model_name in models:
                    try:
                        set_seed(int(seed))
                        model = build_model(model_name, params=dict(model_params_all.get(model_name, {})),
                                            **data_dims)
                        conf_w = float(getattr(model, "confidence_weight", 0.0))
                        train_loader = _loader(x_train_t, pre.y_train, tr_idx, batch_size=spec.batch_size,
                                               shuffle=True, num_workers=spec.num_workers, pin=pin)
                        val_loader = None if val_idx is None else _loader(
                            x_train_t, pre.y_train, val_idx, batch_size=spec.batch_size,
                            shuffle=False, num_workers=spec.num_workers, pin=pin)
                        test_loader = _loader(x_test_t, pre.y_test, test_idx, batch_size=spec.batch_size,
                                              shuffle=False, num_workers=spec.num_workers, pin=pin)

                        train_model(model, train_loader, val_loader, max_epochs=spec.max_epochs,
                                    lr=spec.lr, weight_decay=spec.weight_decay, optimizer=spec.optimizer,
                                    early_stopping_patience=spec.early_stopping_patience, device=dev,
                                    confidence_weight=conf_w, logger=logger)

                        row_tgt_stats = tgt_stats
                        if method == BN_METHOD:
                            n_bn = adapt_bn_statistics(model, test_loader, device=dev, n_passes=1,
                                                       reset=True, cumulative=True)
                            row_tgt_stats = _json({"method": BN_METHOD, "n_bn_layers": int(n_bn),
                                                   "n_passes": 1, "reset": True, "cumulative": True})

                        y_true, y_pred, probs, _ = predict(model, test_loader, device=dev)
                        metrics = evaluate_predictions(y_true, y_pred, probs)

                        cp = ""
                        if save_ckpt and ckpt_dir is not None:
                            ckpt = (Path(ckpt_dir) / method / model_name / f"{task.tag}_seed{seed}.pt")
                            ckpt.parent.mkdir(parents=True, exist_ok=True)
                            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, ckpt)
                            cp = str(ckpt)

                        rows.append({
                            "experiment_id": EXPERIMENT_ID, "method": method, "protocol": task.protocol,
                            "training_scope": task.training_scope, "model": model_name, "seed": int(seed),
                            "subject": task.subject, "train_sessions": task.train_label(),
                            "test_session": task.test_session,
                            "acc": float(metrics["accuracy"]), "bacc": float(metrics["balanced_accuracy"]),
                            "f1": float(metrics["macro_f1"]), "auc": float(metrics["auc"]),
                            "nll": float(metrics["nll"]), "brier": float(metrics["brier"]),
                            "ece": float(metrics["ece"]),
                            "n_train": int(len(tr_idx)),
                            "n_val": 0 if val_idx is None else int(len(val_idx)),
                            "n_test": int(len(test_idx)),
                            "source_alignment_stats": src_stats, "target_alignment_stats": row_tgt_stats,
                            "used_target_x_for_stats": True, "used_target_y_for_training": False,
                            "checkpoint_path": cp, "status": "ok", "error_message": "",
                        })
                        if logger is not None:
                            logger.info("[align] %-22s %-11s %s seed=%d acc=%.4f (ntr=%d nval=%d nte=%d)",
                                        method, model_name, task.tag, int(seed),
                                        float(metrics["accuracy"]), int(len(tr_idx)),
                                        0 if val_idx is None else int(len(val_idx)), int(len(test_idx)))
                    except Exception as exc:
                        rows.append(_fail_row(method, task, model_name, int(seed),
                                              f"train_error: {exc}", used_target_x))
                        if logger is not None:
                            logger.error("[align] %s %s %s seed=%d failed: %s",
                                         task.tag, method, model_name, int(seed), exc)
    return rows
