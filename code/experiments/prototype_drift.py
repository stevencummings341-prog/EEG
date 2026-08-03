"""Phase 2c — Prototype Drift Analysis (frozen-model diagnostic, leakage-free).

用途: 诊断跨 session 掉点是否来自任务表征 (class prototype) 在模型 penultimate
embedding 空间中的漂移。对每个 subject x source_session x target_session x model x
seed: 只在 SOURCE session 上训练 (train + 从 source train 切出的 val 做 early
stopping), TARGET session 仅作 test; 冻结模型后提取 source_train / source_val /
target_test 的 trial-level embedding, 计算 class prototype 与 6 类漂移指标。

输入: processed_manifest.csv 指向的 per-session .npz (status=ok), 三个 baseline 模型。
输出: 每个 (model, seed) 一组 per-run CSV + 每个 (subject, direction) 一个 embedding
      .npz。可读汇总/表/图/报告由 prototype_drift_summarize.py 生成。

NO-LEAKAGE: target labels are used ONLY for offline diagnostic analysis, not for
training or adaptation. n_target_labels_used_for_training is ALWAYS 0.

依赖: numpy>=1.21, torch>=1.12, scikit-learn>=1.0, scipy>=1.7, pandas (汇总阶段)。
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..datasets.session_splits import (
    SessionRecord,
    group_by_subject,
    make_cross_session_pairs,
)
from ..models.registry import build_model
from ..training.trainer import train_model
from ..utils.seed import set_seed
from .session_protocols import (
    TrainSpec,
    _stratified_val,
    load_session_tensors,
)

CLASS_NAMES = {0: "left", 1: "right"}

# --------------------------------------------------------------------------- #
# CSV column orders (shared by runner outputs + summarizer)
# --------------------------------------------------------------------------- #
METRIC_COLUMNS: List[str] = [
    "experiment", "run_id", "dataset", "task", "model", "seed",
    "subject", "source_session", "target_session", "direction",
    "prototype_type", "distance", "embedding_level",
    "acc_source_val", "acc_target", "acc_drop",
    "prototype_drift_left", "prototype_drift_right", "prototype_drift_mean",
    "source_class_separation", "target_class_separation", "separation_change",
    "prototype_direction_cosine",
    "target_margin_mean", "target_margin_std", "target_negative_margin_rate",
    "source_scatter", "target_scatter", "scatter_change",
    "fisher_source", "fisher_target", "fisher_change",
    "n_source_train", "n_source_val", "n_target_test",
    "n_target_labels_used_for_training",
    "used_target_labels_for_training",
    "used_target_labels_for_offline_diagnostic",
    "status",
]

PROTOTYPE_COLUMNS: List[str] = [
    "run_id", "model", "seed", "subject", "source_session", "target_session",
    "direction", "prototype_type", "domain", "class_index", "class_name",
    "embedding_dim", "prototype_l2_norm", "n_trials_used",
    "npz_path", "npz_key", "status",
]

INDEX_COLUMNS: List[str] = [
    "run_id", "model", "seed", "subject", "source_session", "target_session",
    "direction", "split", "n_trials", "embedding_dim",
    "npz_path", "z_key", "y_key", "logits_key", "probs_key", "pred_key", "conf_key",
    "used_target_labels_for_training", "sha256_z",
]

STATUS_COLUMNS: List[str] = [
    "run_id", "model", "seed", "subject", "source_session", "target_session",
    "direction", "status", "n_source_train", "n_source_val", "n_target_test",
    "best_epoch", "acc_source_val", "acc_target", "error_message",
]


# --------------------------------------------------------------------------- #
# Embedding extraction (penultimate features + logits + probs + pred + conf)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    idx: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Dict[str, np.ndarray]:
    """Frozen forward over ``X[idx]`` -> dict of trial-level arrays.

    Returns z (penultimate features [n,d]), logits [n,C], probs [n,C], pred [n],
    conf [n] (learned confidence if present, else fallback = max softmax prob), y [n].
    """
    model.eval()
    model.to(device)
    sel = torch.as_tensor(idx, dtype=torch.long)
    ds = TensorDataset(X[sel], y[sel])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))
    zs: List[np.ndarray] = []
    logits_l: List[np.ndarray] = []
    confs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    has_learned_conf = True
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        out = model(xb)
        feats = out["features"]
        logits = out["logits"]
        zs.append(feats.detach().cpu().numpy())
        logits_l.append(logits.detach().cpu().numpy())
        ys.append(yb.numpy())
        c = out.get("confidence")
        if c is None:
            has_learned_conf = False
        else:
            confs.append(c.detach().cpu().numpy().reshape(-1))
    z = np.concatenate(zs, axis=0).astype(np.float32)
    logits = np.concatenate(logits_l, axis=0).astype(np.float32)
    y_arr = np.concatenate(ys, axis=0).astype(np.int64)
    # Softmax probabilities (numerically stable).
    m = logits.max(axis=1, keepdims=True)
    ex = np.exp(logits - m)
    probs = (ex / ex.sum(axis=1, keepdims=True)).astype(np.float32)
    pred = probs.argmax(axis=1).astype(np.int64)
    if has_learned_conf and confs:
        conf = np.concatenate(confs, axis=0).astype(np.float32)
    else:
        # Fallback confidence = max softmax probability (NOT a learned head).
        conf = probs.max(axis=1).astype(np.float32)
    return {"z": z, "logits": logits, "probs": probs, "pred": pred, "conf": conf, "y": y_arr}


# --------------------------------------------------------------------------- #
# Prototype computation
# --------------------------------------------------------------------------- #
def compute_prototypes(
    z: np.ndarray, y: np.ndarray, pred: np.ndarray, conf: np.ndarray,
    *, ptype: str, n_classes: int = 2,
) -> Dict[str, object]:
    """Class prototypes for one embedding set under one prototype definition.

    Returns dict: {class_index: vector or None, "n": {class: int}, "degenerate": bool}.
    A class is degenerate (vector None) when no trial supports it (e.g. correct_only
    with zero correct trials in that class).
    """
    protos: Dict[int, Optional[np.ndarray]] = {}
    ns: Dict[int, int] = {}
    degenerate = False
    for c in range(n_classes):
        if ptype == "label_based":
            mask = (y == c)
            sub = z[mask]
            n = int(mask.sum())
            proto = sub.mean(axis=0) if n > 0 else None
        elif ptype == "confidence_weighted":
            mask = (y == c)
            sub = z[mask]
            w = conf[mask].astype(np.float64)
            n = int(mask.sum())
            if n > 0 and w.sum() > 1e-12:
                proto = (sub * w[:, None]).sum(axis=0) / w.sum()
            elif n > 0:
                proto = sub.mean(axis=0)
            else:
                proto = None
        elif ptype == "correct_only":
            mask = (y == c) & (pred == c)
            sub = z[mask]
            n = int(mask.sum())
            proto = sub.mean(axis=0) if n > 0 else None
        else:
            raise ValueError(f"unknown prototype_type '{ptype}'")
        protos[c] = None if proto is None else proto.astype(np.float64)
        ns[c] = n
        if proto is None:
            degenerate = True
    return {"protos": protos, "n": ns, "degenerate": degenerate}


# --------------------------------------------------------------------------- #
# Distances + diagnostic metrics
# --------------------------------------------------------------------------- #
def _distance(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    if metric == "euclidean":
        return float(np.linalg.norm(a - b))
    if metric == "cosine":
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return float("nan")
        return float(1.0 - np.dot(a, b) / (na * nb))
    raise ValueError(f"unknown distance '{metric}'")


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _scatter(z: np.ndarray, y: np.ndarray, protos: Dict[int, np.ndarray],
             metric: str, n_classes: int = 2) -> float:
    """Mean distance of each trial to its own class prototype (within-class scatter)."""
    dists: List[float] = []
    for c in range(n_classes):
        mask = (y == c)
        if not mask.any() or protos.get(c) is None:
            continue
        for zi in z[mask]:
            dists.append(_distance(zi, protos[c], metric))
    return float(np.mean(dists)) if dists else float("nan")


def _fisher_ratio(z: np.ndarray, y: np.ndarray, proto0: np.ndarray, proto1: np.ndarray) -> float:
    """1-D Fisher ratio along the prototype-difference direction.

    between = (mean_right_proj - mean_left_proj)^2 ; within = var_left + var_right.
    """
    w = proto1 - proto0
    nw = np.linalg.norm(w)
    if nw < 1e-12:
        return float("nan")
    w = w / nw
    proj = z @ w
    p0 = proj[y == 0]
    p1 = proj[y == 1]
    if len(p0) < 2 or len(p1) < 2:
        return float("nan")
    between = (p1.mean() - p0.mean()) ** 2
    within = p0.var() + p1.var()
    if within < 1e-12:
        return float("nan")
    return float(between / within)


def _margins(z_t: np.ndarray, y_t: np.ndarray, src_protos: Dict[int, np.ndarray],
             metric: str) -> Tuple[float, float, float]:
    """Nearest-source-prototype margin for each target trial.

    margin_i = d(z_i, src_proto[other]) - d(z_i, src_proto[true]).
    Positive => target trial is closer to its own source prototype (correct under
    the source-prototype nearest-centroid rule); negative => misclassified.
    Returns (mean, std, negative_rate).
    """
    margins: List[float] = []
    for zi, yi in zip(z_t, y_t):
        other = 1 - int(yi)
        d_true = _distance(zi, src_protos[int(yi)], metric)
        d_other = _distance(zi, src_protos[other], metric)
        margins.append(d_other - d_true)
    arr = np.asarray(margins, dtype=np.float64)
    neg_rate = float((arr < 0).mean())
    return float(arr.mean()), float(arr.std()), neg_rate


# --------------------------------------------------------------------------- #
# Leakage guards
# --------------------------------------------------------------------------- #
def _assert_no_leakage(source_session: str, target_session: str,
                       tr_idx: np.ndarray, va_idx: Optional[np.ndarray],
                       n_target_labels_used_for_training: int) -> None:
    """Fail-fast guards: source-only train/val; target never in training."""
    assert source_session != target_session, (
        f"LEAKAGE: source_session == target_session ({source_session})")
    # train and val index sets are disjoint within the source session.
    if va_idx is not None and len(va_idx) > 0:
        assert len(set(tr_idx.tolist()) & set(va_idx.tolist())) == 0, (
            "LEAKAGE: source train/val indices overlap")
    assert int(n_target_labels_used_for_training) == 0, (
        "LEAKAGE: target labels must never enter training")


def _finite_or_fail(row: Dict[str, object]) -> None:
    """Fail-fast if a status==ok row carries any NaN/Inf numeric metric."""
    numeric_keys = [
        "acc_source_val", "acc_target", "acc_drop",
        "prototype_drift_left", "prototype_drift_right", "prototype_drift_mean",
        "source_class_separation", "target_class_separation", "separation_change",
        "prototype_direction_cosine",
        "target_margin_mean", "target_margin_std", "target_negative_margin_rate",
        "source_scatter", "target_scatter", "scatter_change",
        "fisher_source", "fisher_target", "fisher_change",
    ]
    for k in numeric_keys:
        v = row.get(k)
        if v is None or v == "":
            continue
        fv = float(v)
        if not np.isfinite(fv):
            raise ValueError(
                f"NaN/Inf in metric '{k}' for {row.get('model')} seed={row.get('seed')} "
                f"{row.get('subject')} {row.get('direction')} "
                f"ptype={row.get('prototype_type')} dist={row.get('distance')}")


def _sha256_of_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# One (subject, directed pair) cell
# --------------------------------------------------------------------------- #
def run_one_cell(
    *,
    subject: str,
    source_session: str,
    target_session: str,
    sess_tensors: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    model_name: str,
    model_params: Dict,
    data_dims: Dict,
    spec: TrainSpec,
    seed: int,
    device: torch.device,
    prototype_types: Sequence[str],
    distances: Sequence[str],
    run_id: str,
    dataset: str,
    task: str,
    embed_dir: Path,
    ckpt_dir: Optional[Path],
    logger=None,
) -> Dict[str, object]:
    """Train (source-only) -> extract embeddings -> prototypes -> metrics.

    Returns dict with keys: metric_rows, prototype_rows, index_rows, status_row.
    """
    set_seed(int(seed))
    Xs, ys = sess_tensors[source_session]
    Xt, yt = sess_tensors[target_session]
    ys_np = ys.numpy()

    all_idx = np.arange(len(ys_np))
    tr_idx, va_idx = _stratified_val(all_idx, ys_np, spec.val_fraction, int(seed))
    if va_idx is None:
        va_idx = np.array([], dtype=int)

    _assert_no_leakage(source_session, target_session, tr_idx, va_idx, 0)

    direction = f"{source_session}->{target_session}"
    status_row: Dict[str, object] = {
        "run_id": run_id, "model": model_name, "seed": int(seed), "subject": subject,
        "source_session": source_session, "target_session": target_session,
        "direction": direction, "status": "ok",
        "n_source_train": int(len(tr_idx)), "n_source_val": int(len(va_idx)),
        "n_target_test": int(len(yt)), "best_epoch": "",
        "acc_source_val": "", "acc_target": "", "error_message": "",
    }

    pin = device.type == "cuda"
    # ----- train on SOURCE only ----- #
    sel_tr = torch.as_tensor(tr_idx, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(Xs[sel_tr], ys[sel_tr]),
                              batch_size=spec.batch_size, shuffle=True,
                              num_workers=spec.num_workers, pin_memory=pin)
    val_loader = None
    if len(va_idx) > 0:
        sel_va = torch.as_tensor(va_idx, dtype=torch.long)
        val_loader = DataLoader(TensorDataset(Xs[sel_va], ys[sel_va]),
                                batch_size=spec.batch_size, shuffle=False,
                                num_workers=spec.num_workers, pin_memory=pin)

    model = build_model(model_name, params=dict(model_params), **data_dims)
    conf_w = float(getattr(model, "confidence_weight", 0.0))
    tinfo = train_model(
        model, train_loader, val_loader,
        max_epochs=spec.max_epochs, lr=spec.lr, weight_decay=spec.weight_decay,
        optimizer=spec.optimizer, early_stopping_patience=spec.early_stopping_patience,
        device=device, confidence_weight=conf_w, logger=logger,
    )
    status_row["best_epoch"] = int(tinfo["best_epoch"])

    if ckpt_dir is not None:
        cp = (Path(ckpt_dir) / model_name /
              f"proto_{subject}_{source_session}-to-{target_session}_seed{seed}.pt")
        cp.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, cp)

    # ----- frozen embedding extraction ----- #
    emb_kwargs = dict(device=device, batch_size=spec.batch_size, num_workers=spec.num_workers)
    src_tr = extract_embeddings(model, Xs, ys, tr_idx, **emb_kwargs)
    src_va = (extract_embeddings(model, Xs, ys, va_idx, **emb_kwargs)
              if len(va_idx) > 0 else None)
    tgt = extract_embeddings(model, Xt, yt, np.arange(len(yt)), **emb_kwargs)
    emb_dim = int(src_tr["z"].shape[1])

    # acc on source val (early-stopping domain) and target test (cross-session drop).
    acc_source_val = (float((src_va["pred"] == src_va["y"]).mean())
                      if src_va is not None and len(src_va["y"]) > 0 else float("nan"))
    acc_target = float((tgt["pred"] == tgt["y"]).mean())
    status_row["acc_source_val"] = round(acc_source_val, 6)
    status_row["acc_target"] = round(acc_target, 6)

    # ----- persist embeddings (npz per cell) ----- #
    npz_path = (Path(embed_dir) / model_name / f"seed{seed}" /
                f"{subject}_{source_session}-to-{target_session}.npz")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, np.ndarray] = {}
    index_rows: List[Dict[str, object]] = []
    splits = {"source_train": src_tr, "source_val": src_va, "target_test": tgt}
    for split_name, blob in splits.items():
        if blob is None:
            continue
        prefix = split_name
        for key in ("z", "y", "logits", "probs", "pred", "conf"):
            payload[f"{prefix}__{key}"] = blob[key]
        index_rows.append({
            "run_id": run_id, "model": model_name, "seed": int(seed), "subject": subject,
            "source_session": source_session, "target_session": target_session,
            "direction": direction, "split": split_name,
            "n_trials": int(len(blob["y"])), "embedding_dim": emb_dim,
            "npz_path": str(npz_path), "z_key": f"{prefix}__z", "y_key": f"{prefix}__y",
            "logits_key": f"{prefix}__logits", "probs_key": f"{prefix}__probs",
            "pred_key": f"{prefix}__pred", "conf_key": f"{prefix}__conf",
            "used_target_labels_for_training": False,
            "sha256_z": _sha256_of_array(blob["z"]),
        })
    np.savez_compressed(npz_path, **payload)

    # ----- prototypes + metrics per (prototype_type, distance) ----- #
    metric_rows: List[Dict[str, object]] = []
    prototype_rows: List[Dict[str, object]] = []
    for ptype in prototype_types:
        src_p = compute_prototypes(src_tr["z"], src_tr["y"], src_tr["pred"], src_tr["conf"],
                                   ptype=ptype, n_classes=data_dims["n_classes"])
        tgt_p = compute_prototypes(tgt["z"], tgt["y"], tgt["pred"], tgt["conf"],
                                   ptype=ptype, n_classes=data_dims["n_classes"])
        # record prototype table rows (vectors stored in npz, table holds metadata).
        for domain, pinfo, blob_prefix in (("source", src_p, "source_train"),
                                           ("target", tgt_p, "target_test")):
            for c in range(data_dims["n_classes"]):
                vec = pinfo["protos"].get(c)
                prototype_rows.append({
                    "run_id": run_id, "model": model_name, "seed": int(seed),
                    "subject": subject, "source_session": source_session,
                    "target_session": target_session, "direction": direction,
                    "prototype_type": ptype, "domain": domain, "class_index": c,
                    "class_name": CLASS_NAMES.get(c, str(c)), "embedding_dim": emb_dim,
                    "prototype_l2_norm": ("" if vec is None else round(float(np.linalg.norm(vec)), 6)),
                    "n_trials_used": int(pinfo["n"].get(c, 0)),
                    "npz_path": str(npz_path), "npz_key": f"{blob_prefix}__z",
                    "status": ("ok" if vec is not None else "degenerate_empty_class"),
                })

        base_row = {
            "experiment": "phase2c_prototype_drift", "run_id": run_id, "dataset": dataset,
            "task": task, "model": model_name, "seed": int(seed), "subject": subject,
            "source_session": source_session, "target_session": target_session,
            "direction": direction, "prototype_type": ptype,
            "embedding_level": "penultimate_embedding",
            "acc_source_val": round(acc_source_val, 6), "acc_target": round(acc_target, 6),
            "acc_drop": round(acc_source_val - acc_target, 6),
            "n_source_train": int(len(tr_idx)), "n_source_val": int(len(va_idx)),
            "n_target_test": int(len(yt)),
            "n_target_labels_used_for_training": 0,
            "used_target_labels_for_training": False,
            "used_target_labels_for_offline_diagnostic": True,
        }

        degenerate = src_p["degenerate"] or tgt_p["degenerate"]
        for dist in distances:
            row = dict(base_row)
            row["distance"] = dist
            if degenerate:
                row["status"] = "degenerate_empty_class"
                # leave metric cells blank (no silent bad numbers).
                metric_rows.append(row)
                continue
            sp0, sp1 = src_p["protos"][0], src_p["protos"][1]
            tp0, tp1 = tgt_p["protos"][0], tgt_p["protos"][1]
            drift_l = _distance(sp0, tp0, dist)
            drift_r = _distance(sp1, tp1, dist)
            src_sep = _distance(sp0, sp1, dist)
            tgt_sep = _distance(tp0, tp1, dist)
            mm, msd, neg = _margins(tgt["z"], tgt["y"], {0: sp0, 1: sp1}, dist)
            src_sc = _scatter(src_tr["z"], src_tr["y"], {0: sp0, 1: sp1}, dist,
                              data_dims["n_classes"])
            tgt_sc = _scatter(tgt["z"], tgt["y"], {0: tp0, 1: tp1}, dist,
                              data_dims["n_classes"])
            f_src = _fisher_ratio(src_tr["z"], src_tr["y"], sp0, sp1)
            f_tgt = _fisher_ratio(tgt["z"], tgt["y"], tp0, tp1)
            row.update({
                "status": "ok",
                "prototype_drift_left": round(drift_l, 6),
                "prototype_drift_right": round(drift_r, 6),
                "prototype_drift_mean": round((drift_l + drift_r) / 2.0, 6),
                "source_class_separation": round(src_sep, 6),
                "target_class_separation": round(tgt_sep, 6),
                "separation_change": round(src_sep - tgt_sep, 6),
                "prototype_direction_cosine": round(_cosine_sim(sp1 - sp0, tp1 - tp0), 6),
                "target_margin_mean": round(mm, 6),
                "target_margin_std": round(msd, 6),
                "target_negative_margin_rate": round(neg, 6),
                "source_scatter": round(src_sc, 6),
                "target_scatter": round(tgt_sc, 6),
                "scatter_change": round(tgt_sc - src_sc, 6),
                "fisher_source": round(f_src, 6),
                "fisher_target": round(f_tgt, 6),
                "fisher_change": round(f_src - f_tgt, 6),
            })
            _finite_or_fail(row)
            metric_rows.append(row)

    return {
        "metric_rows": metric_rows,
        "prototype_rows": prototype_rows,
        "index_rows": index_rows,
        "status_row": status_row,
    }


# --------------------------------------------------------------------------- #
# Full sweep over subjects x directed pairs for one (model, seed)
# --------------------------------------------------------------------------- #
def run_prototype_drift(
    records: Sequence[SessionRecord],
    *,
    model_name: str,
    model_params: Dict,
    data_dims: Dict,
    spec: TrainSpec,
    seed: int,
    device: torch.device,
    prototype_types: Sequence[str],
    distances: Sequence[str],
    run_id: str,
    dataset: str,
    task: str,
    embed_dir: Path,
    ckpt_dir: Optional[Path] = None,
    logger=None,
) -> Dict[str, List[Dict[str, object]]]:
    """Run prototype-drift diagnostic for ONE model + ONE seed over all subjects."""
    by_subj = group_by_subject(records)
    metric_rows: List[Dict[str, object]] = []
    prototype_rows: List[Dict[str, object]] = []
    index_rows: List[Dict[str, object]] = []
    status_rows: List[Dict[str, object]] = []

    for subj, recs in by_subj.items():
        if len(recs) < 2:
            if logger is not None:
                logger.info("[proto] %s has <2 ok sessions; skipped.", subj)
            continue
        sess_tensors = {r.session: load_session_tensors(r.npz_path) for r in recs}
        pairs = make_cross_session_pairs(subj, [r.session for r in recs])
        for pair in pairs:
            src, tgt = pair["train_session"], pair["test_session"]
            try:
                res = run_one_cell(
                    subject=subj, source_session=src, target_session=tgt,
                    sess_tensors=sess_tensors, model_name=model_name,
                    model_params=model_params, data_dims=data_dims, spec=spec,
                    seed=seed, device=device, prototype_types=prototype_types,
                    distances=distances, run_id=run_id, dataset=dataset, task=task,
                    embed_dir=embed_dir, ckpt_dir=ckpt_dir, logger=logger,
                )
                metric_rows.extend(res["metric_rows"])
                prototype_rows.extend(res["prototype_rows"])
                index_rows.extend(res["index_rows"])
                status_rows.append(res["status_row"])
            except Exception as exc:  # record failure; do not fake completion.
                status_rows.append({
                    "run_id": run_id, "model": model_name, "seed": int(seed),
                    "subject": subj, "source_session": src, "target_session": tgt,
                    "direction": f"{src}->{tgt}", "status": "failed",
                    "n_source_train": "", "n_source_val": "", "n_target_test": "",
                    "best_epoch": "", "acc_source_val": "", "acc_target": "",
                    "error_message": f"{type(exc).__name__}: {exc}",
                })
                if logger is not None:
                    logger.error("[proto] FAILED %s %s seed=%d %s->%s: %s",
                                 model_name, subj, seed, src, tgt, exc)
        if logger is not None:
            done = [s for s in status_rows if s["subject"] == subj and s["status"] == "ok"]
            logger.info("[proto] %-11s %s seed=%d cells_ok=%d/%d",
                        model_name, subj, seed, len(done), len(pairs))
    return {
        "metric_rows": metric_rows, "prototype_rows": prototype_rows,
        "index_rows": index_rows, "status_rows": status_rows,
    }


def expected_cells(records: Sequence[SessionRecord]) -> List[Tuple[str, str, str]]:
    """List expected (subject, source, target) directed cells (subjects with >=2 ok)."""
    cells: List[Tuple[str, str, str]] = []
    for subj, recs in group_by_subject(records).items():
        if len(recs) < 2:
            continue
        for pair in make_cross_session_pairs(subj, [r.session for r in recs]):
            cells.append((subj, pair["train_session"], pair["test_session"]))
    return cells
