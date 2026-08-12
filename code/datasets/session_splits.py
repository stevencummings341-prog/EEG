"""Session-level splits for the cross-session DG study (within-CV + cross-session).

This module defines the **exact, leakage-free** data partitioning shared by every
model (EEGNet / DeepConvNet / FBCNet / CAP-EEGNet) so the comparison is fair.
It operates ONLY on the formal preprocessed entry: per-session ``.npz`` listed in
``processed_manifest.csv`` with ``status == ok`` (the 5 failed sessions are never
loaded). It never touches the derivatives ``.mat`` or raw BDF.

Two protocols (see docs/BASELINE_PROTOCOL.md):

  * **within-session**: for ONE (subject, session), Stratified K-fold CV over its
    200 trials. Train and test are disjoint trial sets of the SAME session, so
    there is no cross-session drift — this is the per-session upper bound.
    Stratified => the 100/100 left/right balance is preserved in every fold.

  * **cross-session**: for ONE subject, **directed** session pairs
    ``train_session -> test_session`` (e.g. ses-01->ses-02, ses-02->ses-01, ...),
    keeping only pairs where BOTH sessions are ``ok``. The model trains on all
    trials of ``train_session`` and is tested on all trials of ``test_session``.
    This measures the drop caused by cross-session distribution shift.

Shapes / units / labels:
  * A session ``.npz`` stores ``X`` ``[n_trials, 58, 1000]`` float32 (µV @ 250 Hz)
    and ``y`` ``[n_trials]``. Labels are normalized to **{0,1}** (0=left, 1=right)
    by :func:`normalize_labels`, accepting either {0,1} or {1,2} storage.

Reproducibility: split records are JSON-serializable (fold trial indices for
within; directed session pairs for cross) and saved under
``outputs/<run_id>/splits/`` so any run can be reproduced exactly.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.paths import ses_id, sub_id
from .splits import read_processed_manifest

OK_STATUS = "ok"


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def normalize_labels(y: Sequence[int] | np.ndarray) -> np.ndarray:
    """Map class labels to contiguous ``{0,1,...,C-1}`` as int64.

    Accepts (2-class, historical WBCIC/SHU 2C):
      * already-{0,1}  -> unchanged
      * {1,2}          -> subtract 1  (paper/MATLAB: 1=left, 2=right)
      * any other 2 distinct values -> smaller->0, larger->1

    Accepts (3-class, WBCIC-SHU 3C left/right/foot):
      * already-{0,1,2} -> unchanged
      * {1,2,3}         -> subtract 1

    Raises ValueError on empty input or unsupported label sets so a silently-wrong
    scheme cannot enter training.
    """
    y = np.asarray(y).ravel()
    if y.size == 0:
        raise ValueError("normalize_labels: empty label array.")
    uniq = np.unique(y)
    uset = set(int(v) for v in uniq.tolist())

    if uset <= {0, 1}:
        return y.astype(np.int64)
    if uset <= {1, 2}:
        return (y.astype(np.int64) - 1)
    if uset <= {0, 1, 2}:
        return y.astype(np.int64)
    if uset <= {1, 2, 3}:
        return (y.astype(np.int64) - 1)
    if len(uniq) == 2:
        lo, hi = sorted(int(v) for v in uniq.tolist())
        return np.where(y == lo, 0, 1).astype(np.int64)
    raise ValueError(
        f"normalize_labels: unsupported label set {uset} "
        f"({len(uniq)} distinct). Expected binary {{0,1}}/{{1,2}} or 3-class "
        "{{0,1,2}}/{{1,2,3}} — investigate the npz before training."
    )


# --------------------------------------------------------------------------- #
# OK-session loading from the processed manifest
# --------------------------------------------------------------------------- #
@dataclass
class SessionRecord:
    """One ok session row from processed_manifest.csv (training entry)."""

    subject: str            # 'sub-001'
    session: str            # 'ses-01'
    npz_path: str           # absolute path to the per-session .npz
    n_trials: int
    label_0_count: int
    label_1_count: int

    @property
    def key(self) -> Tuple[str, str]:
        return (self.subject, self.session)


def _to_int(v) -> int:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return 0


def load_ok_sessions(
    manifest_path: str | Path,
    *,
    subjects: Optional[Sequence[str | int]] = None,
    sessions: Optional[Sequence[str | int]] = None,
    status_filter: Sequence[str] = (OK_STATUS,),
    max_subjects: Optional[int] = None,
) -> List[SessionRecord]:
    """Return ok session records from processed_manifest.csv (sorted by subject, session).

    Args:
        manifest_path: processed_manifest.csv (one row per session, has npz_path + status).
        subjects: optional subset, accepts 1 / '1' / 'sub-001'.
        sessions: optional subset, accepts 1 / 'ses-01'.
        status_filter: statuses to keep (default only 'ok').
        max_subjects: cap the number of distinct subjects (after sorting) — used by
            ``--max-subjects`` smoke tests. Applied AFTER the subjects filter.

    Only rows whose status is in ``status_filter`` and whose npz_path is non-empty
    are returned. Raises if the filter yields nothing (so a typo can't silently run
    on zero data).
    """
    rows = read_processed_manifest(manifest_path)
    status_set = set(status_filter) if status_filter else None
    subj_set = {sub_id(s) for s in subjects} if subjects else None
    sess_set = {ses_id(s) for s in sessions} if sessions else None

    records: List[SessionRecord] = []
    for r in rows:
        if status_set and (r.get("status") or "").strip() not in status_set:
            continue
        subj = sub_id(r["subject_id"])
        sess = ses_id(r["session_id"])
        if subj_set and subj not in subj_set:
            continue
        if sess_set and sess not in sess_set:
            continue
        npz = (r.get("npz_path") or "").strip()
        if not npz:
            continue
        records.append(SessionRecord(
            subject=subj, session=sess, npz_path=npz,
            n_trials=_to_int(r.get("n_trials")),
            label_0_count=_to_int(r.get("label_0_count")),
            label_1_count=_to_int(r.get("label_1_count")),
        ))

    records.sort(key=lambda x: (x.subject, x.session))

    if max_subjects is not None and max_subjects > 0:
        kept_subjects = []
        seen = set()
        for rec in records:
            if rec.subject not in seen:
                seen.add(rec.subject)
                kept_subjects.append(rec.subject)
            if len(kept_subjects) >= max_subjects:
                break
        keep = set(kept_subjects)
        records = [rec for rec in records if rec.subject in keep]

    if not records:
        raise ValueError(
            f"No sessions matched in {manifest_path} "
            f"(status={list(status_filter)}, subjects={subjects}, sessions={sessions})."
        )
    return records


def group_by_subject(records: Sequence[SessionRecord]) -> "OrderedDict[str, List[SessionRecord]]":
    """Group ok session records by subject (preserving sorted order)."""
    out: "OrderedDict[str, List[SessionRecord]]" = OrderedDict()
    for rec in records:
        out.setdefault(rec.subject, []).append(rec)
    for subj in out:
        out[subj].sort(key=lambda x: x.session)
    return out


# --------------------------------------------------------------------------- #
# Within-session Stratified K-fold
# --------------------------------------------------------------------------- #
def make_within_session_folds(
    subject: str,
    session: str,
    y: Sequence[int] | np.ndarray,
    n_splits: int = 10,
    seed: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Stratified K-fold (train_idx, test_idx) over the trials of one session.

    Labels are normalized to {0,1} first; stratification preserves the left/right
    balance in every fold (it never breaks the 100/100 balance). ``subject`` /
    ``session`` are used only for clear error messages / provenance.

    Raises ValueError if ``n_splits`` exceeds the smallest per-class trial count
    (which would make a stratified fold impossible) — we never silently change the
    protocol.
    """
    from sklearn.model_selection import StratifiedKFold

    y = normalize_labels(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError(f"{subject}/{session}: only one class present ({classes}); cannot CV.")
    min_count = int(counts.min())
    if n_splits > min_count:
        raise ValueError(
            f"{subject}/{session}: n_splits={n_splits} > smallest class count {min_count}. "
            "Reduce --folds."
        )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr.astype(int), te.astype(int)) for tr, te in skf.split(np.zeros(len(y)), y)]


def build_within_split_record(
    subject: str,
    session: str,
    y: Sequence[int] | np.ndarray,
    n_splits: int,
    seed: int,
) -> Dict[str, object]:
    """JSON-serializable within-session split (fold trial indices) for reproducibility."""
    y_norm = normalize_labels(y)
    folds = make_within_session_folds(subject, session, y_norm, n_splits=n_splits, seed=seed)
    return {
        "protocol": "within_session",
        "subject": subject,
        "session": session,
        "n_trials": int(len(y_norm)),
        "label_dist": {"0": int((y_norm == 0).sum()), "1": int((y_norm == 1).sum())},
        "n_splits": int(n_splits),
        "seed": int(seed),
        "folds": [
            {"fold": k, "train_idx": tr.tolist(), "test_idx": te.tolist()}
            for k, (tr, te) in enumerate(folds)
        ],
    }


# --------------------------------------------------------------------------- #
# Cross-session directed pairs
# --------------------------------------------------------------------------- #
def make_cross_session_pairs(
    subject: str,
    ok_sessions: Sequence[str],
) -> List[Dict[str, str]]:
    """All **directed** (train_session -> test_session) pairs for one subject.

    Only sessions in ``ok_sessions`` are used, so every pair has both train and
    test ``ok`` by construction. For 3 ok sessions this yields 6 directed pairs:
    01->02, 01->03, 02->01, 02->03, 03->01, 03->02. A subject with <2 ok sessions
    yields no pairs.
    """
    ok = sorted(ses_id(s) for s in ok_sessions)
    pairs: List[Dict[str, str]] = []
    for tr in ok:
        for te in ok:
            if tr == te:
                continue
            pairs.append({"subject": subject, "train_session": tr, "test_session": te})
    return pairs


def build_cross_split_record(
    records: Sequence[SessionRecord],
) -> Dict[str, object]:
    """JSON-serializable cross-session plan: directed pairs per subject (both ok)."""
    by_subj = group_by_subject(records)
    all_pairs: List[Dict[str, str]] = []
    per_subject: Dict[str, List[str]] = {}
    for subj, recs in by_subj.items():
        ok_sessions = [r.session for r in recs]
        per_subject[subj] = ok_sessions
        all_pairs.extend(make_cross_session_pairs(subj, ok_sessions))
    return {
        "protocol": "cross_session",
        "directed": True,
        "n_subjects": len(by_subj),
        "ok_sessions_per_subject": per_subject,
        "n_pairs": len(all_pairs),
        "pairs": all_pairs,
    }


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_split_json(obj: Dict[str, object], path: str | Path) -> Path:
    """Write a split record to JSON (parents created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def load_split_json(path: str | Path) -> Dict[str, object]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"split file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
