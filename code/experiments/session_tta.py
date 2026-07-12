"""Phase 3 session TTA experiment — Round-1 scaffold + smoke.

Orchestrates embedding replay, no_tta, minimal T3A, and minimal Oracle
diagnostics. Dispatched via ``code/runners.py`` → ``phase3_tta``.

Round-1 defaults to tiny smoke only (no full sweep).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from code.tta.adapters.embedding_only import EmbeddingOnlyAdapter
from code.tta.eval.metrics import build_result_row, evaluate_predictions
from code.tta.feature_sources.base import make_cell_id
from code.tta.feature_sources.embedding_replay import EmbeddingReplaySource
from code.tta.methods.no_tta import NoTTAMethod
from code.tta.methods.t3a_minimal import MinimalT3AMethod
from code.tta.oracle.base import PROVISIONAL_ORACLE_NOTES
from code.tta.oracle.label_guard import run_label_free, run_oracle
from code.tta.oracle.target_label_proto import TargetLabelProtoOracle
from code.tta.report.smoke_reporter import (
    save_results_csv,
    write_framework_smoke_report,
    write_full_a0_replay_report,
    write_oracle_diagnostic_report,
    write_replay_validation_report,
    write_t3a_smoke_report,
)
from code.utils.logging_utils import get_logger

logger = get_logger("experiments.session_tta")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _abs(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def select_smoke_subjects(
    drift_summary_csv: Path,
    *,
    embeddings_dir: Path,
    model: str,
    seed: int,
    n_per_level: int = 1,
) -> pd.DataFrame:
    """Auto-select 1 stable + 1 high-drift subject that have embeddings (hard constraint #3)."""
    if not drift_summary_csv.is_file():
        raise FileNotFoundError(
            f"drift summary not found for auto subject selection: {drift_summary_csv}"
        )
    df = pd.read_csv(drift_summary_csv)
    if "drift_level" not in df.columns or "subject" not in df.columns:
        raise ValueError(f"drift summary missing columns: {drift_summary_csv}")

    rows: List[Dict[str, Any]] = []
    for level in ("stable", "high"):
        cand = df[df["drift_level"].astype(str).str.lower() == level].copy()
        if "drift_score" in cand.columns:
            # stable: lowest score first; high: highest score first
            ascending = level == "stable"
            cand = cand.sort_values("drift_score", ascending=ascending)
        picked = 0
        for _, r in cand.iterrows():
            subj = str(r["subject"])
            # Prefer ses-01->ses-02 if present
            npz = (
                embeddings_dir
                / model
                / f"seed{int(seed)}"
                / f"{subj}_ses-01-to-ses-02.npz"
            )
            if not npz.is_file():
                # any npz for this subject?
                folder = embeddings_dir / model / f"seed{int(seed)}"
                matches = sorted(folder.glob(f"{subj}_*.npz")) if folder.is_dir() else []
                if not matches:
                    continue
                npz = matches[0]
                # parse sessions from filename {subj}_{src}-to-{tgt}.npz
                stem = npz.stem[len(subj) + 1 :]
                src, tgt = stem.split("-to-")
            else:
                src, tgt = "ses-01", "ses-02"
            rows.append(
                {
                    "subject": subj,
                    "drift_level": level,
                    "drift_score": float(r.get("drift_score", float("nan"))),
                    "source_session": src,
                    "target_session": tgt,
                    "selection_rule": "auto_from_drift_tertile+embedding_exists",
                    "npz_exists": True,
                }
            )
            picked += 1
            if picked >= n_per_level:
                break
        if picked < n_per_level:
            raise RuntimeError(
                f"auto subject selection failed for drift_level={level}: "
                f"needed {n_per_level}, found {picked}. "
                f"Check {drift_summary_csv} and embeddings under {embeddings_dir}/{model}/seed{seed}/"
            )
    return pd.DataFrame(rows)


def _load_phase2c_acc_map(
    metrics_csv: Path,
    *,
    dataset: str,
    model: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Map cell_id -> Phase 2c acc_target (label_based/euclidean preferred if present)."""
    if not metrics_csv.is_file():
        return {}
    df = pd.read_csv(metrics_csv)
    if "prototype_type" in df.columns:
        pref = df[df["prototype_type"].astype(str) == "label_based"]
        if len(pref):
            df = pref
    if "distance" in df.columns:
        pref = df[df["distance"].astype(str) == "euclidean"]
        if len(pref):
            df = pref
    if model is not None and "model" in df.columns:
        df = df[df["model"].astype(str) == model]
    if seed is not None and "seed" in df.columns:
        df = df[df["seed"].astype(int) == int(seed)]
    out: Dict[str, float] = {}
    for _, r in df.iterrows():
        cid = make_cell_id(
            dataset,
            str(r["model"]),
            int(r["seed"]),
            str(r["subject"]),
            str(r["source_session"]),
            str(r["target_session"]),
        )
        # Prefer first occurrence (already filtered)
        if cid not in out:
            out[cid] = float(r["acc_target"])
    return out


def _parse_embedding_filename(npz_path: Path) -> Optional[Tuple[str, str, str]]:
    """Parse ``{subject}_{source_session}-to-{target_session}.npz`` -> (subj, src, tgt)."""
    stem = npz_path.stem
    if "_" not in stem or "-to-" not in stem:
        return None
    subject, rest = stem.split("_", 1)
    if "-to-" not in rest:
        return None
    src, tgt = rest.split("-to-", 1)
    if not subject or not src or not tgt:
        return None
    return subject, src, tgt


def build_canonical_universe(
    *,
    dataset: str,
    metrics_csv: Path,
    index_glob: str,
    embeddings_dir: Path,
    models: Sequence[str],
    seeds: Sequence[int],
    root: Path,
) -> Dict[str, Any]:
    """Build the canonical Phase 2c A0 cell universe and cross-check consistency.

    Sources checked (all read-only, never trusting stale npz_path columns):
      1. Phase 2c metrics (``prototype_type==label_based`` & ``distance==euclidean``
         preferred, matching ``_load_phase2c_acc_map``) — the reference `acc_target`.
      2. Phase 2c embed_index__*.csv files — index of what was *supposed* to be dumped.
      3. Actual embedding npz files on disk under ``embeddings_dir``.

    Returns a dict with per-source cell_id sets, duplicate/missing/unexpected
    diagnostics, and the canonical cell list (metrics ∩ embeddings, with
    resolved paths + expected acc) that full_a0_replay will actually replay.
    """
    models_set = {str(m) for m in models}
    seeds_set = {int(s) for s in seeds}

    if not metrics_csv.is_file():
        raise FileNotFoundError(f"Phase 2c metrics not found: {metrics_csv}")
    mdf = pd.read_csv(metrics_csv)
    for col, val in (("prototype_type", "label_based"), ("distance", "euclidean")):
        if col in mdf.columns:
            pref = mdf[mdf[col].astype(str) == val]
            if len(pref):
                mdf = pref
    mdf = mdf[
        mdf["model"].astype(str).isin(models_set) & mdf["seed"].astype(int).isin(seeds_set)
    ].copy()
    mdf["cell_id"] = [
        make_cell_id(
            dataset, str(r.model), int(r.seed), str(r.subject),
            str(r.source_session), str(r.target_session),
        )
        for r in mdf.itertuples()
    ]
    metrics_dupe_counts = mdf["cell_id"].value_counts()
    metrics_duplicates = sorted(metrics_dupe_counts[metrics_dupe_counts > 1].index.tolist())
    mdf_dedup = mdf.drop_duplicates(subset=["cell_id"], keep="first").set_index("cell_id")
    metrics_ids = set(mdf_dedup.index)
    acc_map: Dict[str, float] = mdf_dedup["acc_target"].astype(float).to_dict()
    n_target_metrics_map: Dict[str, Any] = (
        mdf_dedup["n_target_test"].to_dict() if "n_target_test" in mdf_dedup.columns else {}
    )

    index_files_found = sorted(root.glob(index_glob))
    idx_frames: List[pd.DataFrame] = []
    index_files_used: List[Path] = []
    for f in index_files_found:
        stem = f.stem  # e.g. embed_index__eegnet__seed0
        parts = stem.split("__")
        if len(parts) != 3 or not parts[2].startswith("seed"):
            continue
        m = parts[1]
        try:
            s = int(parts[2][len("seed"):])
        except ValueError:
            continue
        if m not in models_set or s not in seeds_set:
            continue
        idx_frames.append(pd.read_csv(f))
        index_files_used.append(f)
    idx_df = pd.concat(idx_frames, ignore_index=True) if idx_frames else pd.DataFrame()

    index_ids: set = set()
    index_duplicate_cells: List[str] = []
    incomplete_split_cells: List[str] = []
    if len(idx_df):
        idx_df["cell_id"] = [
            make_cell_id(
                dataset, str(r.model), int(r.seed), str(r.subject),
                str(r.source_session), str(r.target_session),
            )
            for r in idx_df.itertuples()
        ]
        index_ids = set(idx_df["cell_id"].unique())
        if "split" in idx_df.columns:
            dup_mask = idx_df.duplicated(subset=["cell_id", "split"], keep=False)
            index_duplicate_cells = sorted(idx_df.loc[dup_mask, "cell_id"].unique().tolist())
            expected_splits = {"source_train", "source_val", "target_test"}
            split_sets = idx_df.groupby("cell_id")["split"].apply(lambda s: frozenset(s))
            incomplete_split_cells = sorted(
                cid for cid, sset in split_sets.items() if set(sset) != expected_splits
            )

    emb_path_map: Dict[str, Path] = {}
    embedding_duplicate_cells: List[str] = []
    for m in sorted(models_set):
        for s in sorted(seeds_set):
            folder = embeddings_dir / m / f"seed{s}"
            if not folder.is_dir():
                continue
            for npz in sorted(folder.glob("*.npz")):
                parsed = _parse_embedding_filename(npz)
                if parsed is None:
                    continue
                subj, src, tgt = parsed
                cid = make_cell_id(dataset, m, s, subj, src, tgt)
                if cid in emb_path_map:
                    embedding_duplicate_cells.append(cid)
                emb_path_map[cid] = npz.resolve()
    emb_ids = set(emb_path_map.keys())

    missing_embedding = sorted(metrics_ids - emb_ids)
    unexpected_embedding = sorted(emb_ids - metrics_ids)
    missing_from_index = sorted(metrics_ids - index_ids) if index_ids else []
    unexpected_in_index = sorted(index_ids - metrics_ids) if index_ids else []

    canonical_ids = sorted(metrics_ids & emb_ids)
    canonical_cells: List[Dict[str, Any]] = []
    for cid in canonical_ids:
        row = mdf_dedup.loc[cid]
        canonical_cells.append(
            {
                "cell_id": cid,
                "model": str(row["model"]),
                "seed": int(row["seed"]),
                "subject": str(row["subject"]),
                "source_session": str(row["source_session"]),
                "target_session": str(row["target_session"]),
                "direction": str(row.get("direction", "")),
                "acc_target_expected": float(row["acc_target"]),
                "n_target_expected": n_target_metrics_map.get(cid),
                "npz_path_resolved": str(emb_path_map[cid]),
            }
        )

    return {
        "n_metrics_cells": len(metrics_ids),
        "n_index_cells": len(index_ids),
        "n_embedding_cells": len(emb_ids),
        "n_canonical_valid": len(canonical_ids),
        "metrics_duplicates": metrics_duplicates,
        "index_duplicate_cells": index_duplicate_cells,
        "incomplete_split_cells": incomplete_split_cells,
        "embedding_duplicate_cells": sorted(set(embedding_duplicate_cells)),
        "missing_embedding": missing_embedding,
        "unexpected_embedding": unexpected_embedding,
        "missing_from_index": missing_from_index,
        "unexpected_in_index": unexpected_in_index,
        "canonical_cells": canonical_cells,
        "acc_map": acc_map,
        "index_files_found": [str(f) for f in index_files_found],
        "index_files_used": [str(f) for f in index_files_used],
    }


def run_full_a0_replay(
    cfg: Dict[str, Any], *, root: Path, dataset: str
) -> Dict[str, Any]:
    """Opt-in full-universe no_tta embedding replay validation against Phase 2c.

    Builds the canonical cell universe dynamically from Phase 2c index/metrics
    (no hardcoded cell counts), replays no_tta only for every valid cell, and
    reports match/mismatch + |Δ| stats. Does NOT run t3a/oracle by default and
    does NOT expand any method matrix — this is a replay/consistency check.
    """
    round1 = cfg.get("round1") or {}
    a0_cfg = round1.get("full_a0_replay") or {}
    tolerance = float(a0_cfg.get("tolerance", 1e-6))

    models = list(a0_cfg.get("models") or cfg.get("models") or ["eegnet"])
    seeds = list(a0_cfg.get("seeds") or cfg.get("seeds") or [0])

    src_emb = cfg.get("source_embeddings") or {}
    embeddings_dir = _abs(
        a0_cfg.get(
            "embeddings_dir",
            src_emb.get("embeddings_dir", f"outputs/experiments/{dataset}/prototype_drift_v1/embeddings"),
        )
    )
    index_glob = str(
        a0_cfg.get("index_glob", src_emb.get("index_glob", f"outputs/experiments/{dataset}/prototype_drift_v1/runs/embed_index__*.csv"))
    )
    metrics_csv = _abs(
        a0_cfg.get(
            "phase2c_metrics",
            f"4_experiments/{dataset}/prototype_drift/tables/prototype_drift_metrics.csv",
        )
    )

    out_cfg = cfg.get("output") or {}
    readable = _abs(out_cfg.get("readable_dir", f"4_experiments/{dataset}/tta"))
    run_dir = _abs(out_cfg.get("run_dir", out_cfg.get("output_dir", f"outputs/experiments/{dataset}/tta_v1")))
    replay_dir = readable / "replay_validation"
    for d in (readable, replay_dir, run_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger.info(
        "full_a0_replay: building canonical universe (dataset=%s models=%s seeds=%s)",
        dataset, models, seeds,
    )
    universe = build_canonical_universe(
        dataset=dataset,
        metrics_csv=metrics_csv,
        index_glob=index_glob,
        embeddings_dir=embeddings_dir,
        models=models,
        seeds=seeds,
        root=root,
    )
    canonical_cells = universe["canonical_cells"]
    logger.info(
        "full_a0_replay: universe built — metrics=%d index=%d embeddings=%d canonical_valid=%d",
        universe["n_metrics_cells"], universe["n_index_cells"],
        universe["n_embedding_cells"], universe["n_canonical_valid"],
    )

    # --- write consistency (mismatch) table up front, before the (potentially
    # slow) replay loop, so partial/blocked verdicts are visible even if the
    # replay loop is interrupted. ---
    mismatch_rows: List[Dict[str, Any]] = []
    for cid in universe["metrics_duplicates"]:
        mismatch_rows.append({"cell_id": cid, "issue": "metrics_duplicate"})
    for cid in universe["index_duplicate_cells"]:
        mismatch_rows.append({"cell_id": cid, "issue": "index_duplicate_split_row"})
    for cid in universe["incomplete_split_cells"]:
        mismatch_rows.append({"cell_id": cid, "issue": "index_incomplete_split_set"})
    for cid in universe["embedding_duplicate_cells"]:
        mismatch_rows.append({"cell_id": cid, "issue": "embedding_duplicate"})
    for cid in universe["missing_embedding"]:
        mismatch_rows.append({"cell_id": cid, "issue": "missing_embedding"})
    for cid in universe["unexpected_embedding"]:
        mismatch_rows.append({"cell_id": cid, "issue": "unexpected_embedding"})
    for cid in universe["missing_from_index"]:
        mismatch_rows.append({"cell_id": cid, "issue": "missing_from_index"})
    for cid in universe["unexpected_in_index"]:
        mismatch_rows.append({"cell_id": cid, "issue": "unexpected_in_index"})
    save_results_csv(replay_dir / "full_a0_universe_consistency.csv", mismatch_rows)

    n_cells = len(canonical_cells)
    max_cells_cap = a0_cfg.get("max_cells")  # optional safety cap for testing only
    if max_cells_cap is not None:
        canonical_cells = canonical_cells[: int(max_cells_cap)]
        logger.info(
            "full_a0_replay: max_cells cap active (%s) — replaying %d/%d canonical cells",
            max_cells_cap, len(canonical_cells), n_cells,
        )

    source = EmbeddingReplaySource(embeddings_dir=embeddings_dir, dataset=dataset, project_root=root)
    no_tta = NoTTAMethod()

    replay_rows: List[Dict[str, Any]] = []
    t0 = time.time()
    log_every = max(1, int(a0_cfg.get("log_every", 250)))
    for i, spec in enumerate(canonical_cells, start=1):
        row: Dict[str, Any] = {
            "cell_id": spec["cell_id"],
            "model": spec["model"],
            "seed": spec["seed"],
            "subject": spec["subject"],
            "source_session": spec["source_session"],
            "target_session": spec["target_session"],
            "direction": spec.get("direction", f"{spec['source_session']}->{spec['target_session']}"),
            "acc_target_expected": spec["acc_target_expected"],
            "n_target_expected": spec.get("n_target_expected"),
            "npz_path_resolved": spec["npz_path_resolved"],
        }
        try:
            bundle = source.load_cell(
                model=spec["model"],
                seed=spec["seed"],
                subject=spec["subject"],
                source_session=spec["source_session"],
                target_session=spec["target_session"],
            )
            r0 = run_label_free(no_tta, bundle)
            if bundle.target_y_true is not None and r0.pred is not None:
                metrics = evaluate_predictions(bundle.target_y_true, r0.pred)
                replay_acc = metrics["acc"]
            else:
                replay_acc = float("nan")
            delta = (
                float(replay_acc - spec["acc_target_expected"])
                if np.isfinite(replay_acc)
                else float("nan")
            )
            row.update(
                {
                    "n_target_actual": bundle.n_target,
                    "replay_acc": replay_acc,
                    "delta": delta,
                    "abs_delta": abs(delta) if np.isfinite(delta) else float("nan"),
                    "status": "ok" if np.isfinite(delta) else "eval_failed",
                    "pass_tol": bool(np.isfinite(delta) and abs(delta) < tolerance),
                    "failure_reason": "",
                }
            )
        except Exception as e:  # noqa: BLE001 - record and continue full sweep
            row.update(
                {
                    "n_target_actual": None,
                    "replay_acc": float("nan"),
                    "delta": float("nan"),
                    "abs_delta": float("nan"),
                    "status": "load_failed",
                    "pass_tol": False,
                    "failure_reason": str(e),
                }
            )
        replay_rows.append(row)
        if i % log_every == 0 or i == len(canonical_cells):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else float("nan")
            logger.info(
                "full_a0_replay: %d/%d cells done (%.1fs elapsed, %.1f cells/s)",
                i, len(canonical_cells), elapsed, rate,
            )
    elapsed_sec = round(time.time() - t0, 1)

    save_results_csv(replay_dir / "full_a0_replay_delta_table.csv", replay_rows)

    n_attempted = len(replay_rows)
    n_ok = sum(1 for r in replay_rows if r["status"] == "ok")
    n_failed = n_attempted - n_ok
    n_pass = sum(1 for r in replay_rows if r["pass_tol"])
    n_fail_tolerance = n_ok - n_pass
    abs_vals = [r["abs_delta"] for r in replay_rows if np.isfinite(r.get("abs_delta", float("nan")))]
    max_abs_delta = float(max(abs_vals)) if abs_vals else None

    by_rows = pd.DataFrame(replay_rows)
    by_summary: List[Dict[str, Any]] = []
    if len(by_rows):
        grouped = by_rows.groupby(["model", "seed", "direction"], dropna=False)
        for (m, s, d), g in grouped:
            g_abs = g["abs_delta"][np.isfinite(g["abs_delta"])]
            by_summary.append(
                {
                    "model": m,
                    "seed": int(s),
                    "direction": d,
                    "n": int(len(g)),
                    "n_pass": int(g["pass_tol"].sum()),
                    "n_fail": int((~g["pass_tol"]).sum()),
                    "max_abs_delta": float(g_abs.max()) if len(g_abs) else None,
                }
            )
        by_summary.sort(key=lambda r: (r["max_abs_delta"] if r["max_abs_delta"] is not None else -1), reverse=True)
    save_results_csv(replay_dir / "full_a0_summary_by_model_seed_direction.csv", by_summary)

    sets_consistent = not any(
        [
            universe["metrics_duplicates"],
            universe["index_duplicate_cells"],
            universe["incomplete_split_cells"],
            universe["embedding_duplicate_cells"],
            universe["missing_embedding"],
            universe["unexpected_embedding"],
            universe["missing_from_index"],
            universe["unexpected_in_index"],
        ]
    )
    all_replayed_ok = n_failed == 0
    all_pass_tol = n_ok > 0 and n_fail_tolerance == 0
    capped = max_cells_cap is not None

    if universe["n_canonical_valid"] == 0 or universe["n_metrics_cells"] == 0:
        verdict = "blocked"
    elif capped:
        # Intentionally partial cap (e.g. dev smoke-test of full_a0_replay itself).
        verdict = "partial"
    elif sets_consistent and all_replayed_ok and all_pass_tol:
        verdict = "complete"
    else:
        verdict = "partial"

    replay_summary = {
        "n_attempted": n_attempted,
        "n_ok": n_ok,
        "n_failed": n_failed,
        "n_pass": n_pass,
        "n_fail_tolerance": n_fail_tolerance,
        "tolerance": tolerance,
        "max_abs_delta": max_abs_delta,
        "elapsed_sec": elapsed_sec,
        "by_model_seed_direction": by_summary,
    }

    notes = [
        f"models={models}, seeds={seeds}, tolerance={tolerance}",
        "no_tta replay only; T3A/Tent/SHOT/Oracle NOT run in full_a0_replay mode.",
        f"index files used: {len(universe['index_files_used'])}",
    ]
    if capped:
        notes.append(
            f"max_cells cap active ({max_cells_cap}); verdict forced to 'partial' "
            "(this is a dev-test subset run, not the real full A0)."
        )

    write_full_a0_replay_report(
        readable / "reports" / "FULL_A0_REPLAY_VALIDATION_REPORT.md",
        verdict=verdict,
        dataset=dataset,
        universe=universe,
        replay=replay_summary,
        notes=notes,
    )

    summary: Dict[str, Any] = {
        "mode": "full_a0_replay",
        "dataset": dataset,
        "models": models,
        "seeds": seeds,
        "status": verdict,
        "verdict": verdict,
        "dry_run": False,
        "provisional_oracle_note": PROVISIONAL_ORACLE_NOTES,
        "universe": {k: v for k, v in universe.items() if k not in ("canonical_cells", "acc_map")},
        "replay": {k: v for k, v in replay_summary.items() if k != "by_model_seed_direction"},
        "by_model_seed_direction_top5": by_summary[:5],
        "readable_dir": str(readable),
        "replay_delta_table": str(replay_dir / "full_a0_replay_delta_table.csv"),
        "universe_consistency_table": str(replay_dir / "full_a0_universe_consistency.csv"),
        "report_md": str(readable / "reports" / "FULL_A0_REPLAY_VALIDATION_REPORT.md"),
    }
    (run_dir / "full_a0_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    logger.info("full_a0_replay done: verdict=%s max_abs_delta=%s n_ok=%d/%d", verdict, max_abs_delta, n_ok, n_attempted)
    return summary


def run_cell_methods(
    bundle,
    *,
    model_adapter_name: str,
    t3a_cfg: Dict[str, Any],
    run_oracle_flag: bool,
    n_classes: int = 2,
    run_t3a_flag: bool = True,
) -> Tuple[List[Dict[str, Any]], Optional[float]]:
    rows: List[Dict[str, Any]] = []
    no_tta = NoTTAMethod()
    r0 = run_label_free(no_tta, bundle)
    m0 = build_result_row(
        bundle=bundle, result=r0, model_adapter=model_adapter_name, no_tta_acc=None
    )
    no_tta_acc = float(m0["acc"]) if np.isfinite(m0["acc"]) else None
    m0["delta_vs_no_tta"] = 0.0 if no_tta_acc is not None else float("nan")
    m0["negative_transfer"] = False
    rows.append(m0)

    if run_t3a_flag:
        t3a = MinimalT3AMethod(
            geometry=str(t3a_cfg.get("geometry", "cosine")),
            filter_k=int(t3a_cfg.get("filter_k", 20)),
            initialization=str(t3a_cfg.get("initialization", "src_proto")),
            temperature=float(t3a_cfg.get("temperature", 1.0)),
            episodic=bool(t3a_cfg.get("episodic", False)),
            n_classes=n_classes,
            seed=int(bundle.seed),
        )
        r1 = run_label_free(t3a, bundle)
        rows.append(
            build_result_row(
                bundle=bundle,
                result=r1,
                model_adapter=model_adapter_name,
                no_tta_acc=no_tta_acc,
            )
        )

    if run_oracle_flag:
        oracle = TargetLabelProtoOracle(
            geometry=str(t3a_cfg.get("oracle_geometry", t3a_cfg.get("geometry", "cosine"))),
            n_classes=n_classes,
        )
        r2 = run_oracle(oracle, bundle)
        rows.append(
            build_result_row(
                bundle=bundle,
                result=r2,
                model_adapter=model_adapter_name,
                no_tta_acc=no_tta_acc,
            )
        )
    return rows, no_tta_acc


def run_session_tta(cfg: Dict[str, Any], *, project_root: Optional[Path] = None) -> Dict[str, Any]:
    """Main Round-1 entry used by the phase3_tta runner.

    ``round1.mode`` selects the behavior:
      - ``smoke`` (default): tiny auto-selected subject smoke (existing Round-1 behavior).
      - ``full_a0_replay``: opt-in full-universe no_tta replay validation against
        Phase 2c (see :func:`run_full_a0_replay`). Must be explicitly requested
        in config; never the default.
    """
    root = Path(project_root) if project_root else PROJECT_ROOT
    round1 = cfg.get("round1") or {}
    mode = str(round1.get("mode", cfg.get("mode", "smoke"))).lower()
    dry_run = bool(round1.get("dry_run", False))

    dataset = str(cfg.get("dataset") or (cfg.get("data") or {}).get("name") or "wbci_shu")

    if dry_run:
        # Hard constraint: dry_run must not create directories or write files.
        logger.info(
            "dry_run=True (mode=%s, dataset=%s): no I/O performed, no directories created.",
            mode, dataset,
        )
        return {
            "mode": mode,
            "dataset": dataset,
            "dry_run": True,
            "status": "dry_run",
            "provisional_oracle_note": PROVISIONAL_ORACLE_NOTES,
        }

    if mode == "full_a0_replay":
        return run_full_a0_replay(cfg, root=root, dataset=dataset)

    max_cells = int(round1.get("max_cells", 4))
    run_oracle_flag = bool(round1.get("run_oracle", True))
    # Default True preserves existing WBCIC Round-1 smoke behavior (no_tta +
    # t3a_minimal + optional oracle). SHU smoke opts out via run_t3a: false
    # (task spec: SHU minimal smoke is no_tta only).
    run_t3a_flag = bool(round1.get("run_t3a", True))
    do_replay = bool(round1.get("run_replay_validation", True))

    src_emb = cfg.get("source_embeddings") or {}
    embeddings_dir = _abs(src_emb.get("embeddings_dir", f"outputs/experiments/{dataset}/prototype_drift_v1/embeddings"))
    if not embeddings_dir.is_absolute():
        embeddings_dir = root / embeddings_dir

    out_cfg = cfg.get("output") or {}
    readable = _abs(out_cfg.get("readable_dir", f"4_experiments/{dataset}/tta"))
    run_dir = _abs(out_cfg.get("run_dir", out_cfg.get("output_dir", f"outputs/experiments/{dataset}/tta_v1")))
    for d in (
        readable,
        readable / "smoke",
        readable / "replay_validation",
        readable / "oracle_diagnostic",
        readable / "method_catalog",
        readable / "reports",
        readable / "tables",
        readable / "figures",
        run_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)

    smoke_cfg = round1.get("smoke") or {}
    model = str(smoke_cfg.get("model", (cfg.get("models") or ["eegnet"])[0]))
    seed = int(smoke_cfg.get("seed", (cfg.get("seeds") or [0])[0]))
    # model_adapter name for schema: example adapter label, not a backend hard-bind
    model_adapter_name = str(smoke_cfg.get("model_adapter", model))

    t3a_smoke = smoke_cfg.get("t3a") or {
        "geometry": "cosine",
        "filter_k": 20,
        "initialization": "src_proto",
        "episodic": False,
        "temperature": 1.0,
    }
    n_classes = int((cfg.get("data") or {}).get("n_classes", 2))

    drift_csv = _abs(
        smoke_cfg.get(
            "drift_summary",
            f"1_session_drift/{dataset}/tables/per_subject_drift_summary.csv",
        )
    )
    metrics_csv = _abs(
        smoke_cfg.get(
            "phase2c_metrics",
            f"4_experiments/{dataset}/prototype_drift/tables/prototype_drift_metrics.csv",
        )
    )

    summary: Dict[str, Any] = {
        "mode": mode,
        "dataset": dataset,
        "model": model,
        "model_adapter": model_adapter_name,
        "seed": seed,
        "dry_run": dry_run,
        "run_t3a": run_t3a_flag,
        "run_oracle": run_oracle_flag,
        "provisional_oracle_note": PROVISIONAL_ORACLE_NOTES,
    }

    # --- select cells ---
    selected = select_smoke_subjects(
        drift_csv, embeddings_dir=embeddings_dir, model=model, seed=seed
    )
    selected_path = readable / "smoke" / "selected_smoke_cells.csv"
    selected.to_csv(selected_path, index=False)
    logger.info("selected smoke subjects -> %s", selected_path)

    # limit cells
    cell_specs: List[Dict[str, Any]] = []
    for _, r in selected.iterrows():
        cell_specs.append(
            {
                "subject": r["subject"],
                "source_session": r["source_session"],
                "target_session": r["target_session"],
                "drift_level": r["drift_level"],
            }
        )
    # Optionally add a second direction for the first subject if max_cells allows
    if max_cells > len(cell_specs) and cell_specs:
        s0 = cell_specs[0]
        alt = {
            "subject": s0["subject"],
            "source_session": "ses-01",
            "target_session": "ses-03",
            "drift_level": s0["drift_level"],
        }
        npz_alt = (
            embeddings_dir
            / model
            / f"seed{seed}"
            / f"{alt['subject']}_{alt['source_session']}-to-{alt['target_session']}.npz"
        )
        if npz_alt.is_file():
            cell_specs.append(alt)
    cell_specs = cell_specs[:max_cells]

    source = EmbeddingReplaySource(
        embeddings_dir=embeddings_dir, dataset=dataset, project_root=root
    )
    # embedding-only adapter for metadata (no live forward in Round-1 smoke)
    _adapter = EmbeddingOnlyAdapter(n_classes=n_classes, metadata={"example_model_name": model})

    all_rows: List[Dict[str, Any]] = []
    replay_rows: List[Dict[str, Any]] = []
    oracle_rows: List[Dict[str, Any]] = []
    phase2c_acc = _load_phase2c_acc_map(metrics_csv, dataset=dataset, model=model, seed=seed)

    for spec in cell_specs:
        try:
            bundle = source.load_cell(
                model=model,
                seed=seed,
                subject=spec["subject"],
                source_session=spec["source_session"],
                target_session=spec["target_session"],
            )
            if bundle.embedding_dim and _adapter._feature_dim is None:
                _adapter._feature_dim = bundle.embedding_dim
        except Exception as e:
            logger.error("load_cell failed: %s", e)
            fail_cid = make_cell_id(
                dataset, model, seed, spec["subject"],
                spec["source_session"], spec["target_session"],
            )
            all_rows.append(
                {
                    "dataset": dataset,
                    "model_adapter": model_adapter_name,
                    "method": "load_failed",
                    "seed": seed,
                    "subject": spec["subject"],
                    "source_session": spec["source_session"],
                    "target_session": spec["target_session"],
                    "cell_id": fail_cid,
                    "feature_source": "embedding_replay",
                    "n_source": 0,
                    "n_target": 0,
                    "acc": float("nan"),
                    "balanced_acc": float("nan"),
                    "delta_vs_no_tta": float("nan"),
                    "negative_transfer": False,
                    "used_target_labels": False,
                    "oracle_diagnostic_only": False,
                    "not_deployable": False,
                    "geometry": "",
                    "filter_k": "",
                    "initialization": "",
                    "npz_path_resolved": "",
                    "failure_reason": str(e),
                }
            )
            continue

        rows, no_tta_acc = run_cell_methods(
            bundle,
            model_adapter_name=model_adapter_name,
            t3a_cfg=t3a_smoke,
            run_oracle_flag=run_oracle_flag,
            n_classes=n_classes,
            run_t3a_flag=run_t3a_flag,
        )
        all_rows.extend(rows)
        for row in rows:
            if row["method"] == "no_tta":
                ref = phase2c_acc.get(bundle.cell_id)
                delta = (
                    float(row["acc"] - ref)
                    if ref is not None and np.isfinite(row["acc"])
                    else float("nan")
                )
                replay_rows.append(
                    {
                        "cell_id": bundle.cell_id,
                        "npz_path_resolved": bundle.npz_path_resolved,
                        "n_target_trials": bundle.n_target,
                        "no_tta_acc": row["acc"],
                        "phase2c_acc_target": ref if ref is not None else float("nan"),
                        "abs_delta": abs(delta) if np.isfinite(delta) else float("nan"),
                        "pass_1e-6": bool(np.isfinite(delta) and abs(delta) < 1e-6),
                    }
                )
            if row.get("oracle_diagnostic_only"):
                oracle_rows.append(row)

    # --- write outputs ---
    smoke_csv = readable / "smoke" / "smoke_results.csv"
    save_results_csv(smoke_csv, all_rows)
    save_results_csv(run_dir / "smoke_results.csv", all_rows)

    # replay validation
    n_pass = sum(1 for r in replay_rows if r.get("pass_1e-6"))
    n_fail = len(replay_rows) - n_pass
    max_abs = None
    abs_vals = [r["abs_delta"] for r in replay_rows if np.isfinite(r.get("abs_delta", float("nan")))]
    if abs_vals:
        max_abs = float(max(abs_vals))
    diagnosis = []
    if not phase2c_acc:
        diagnosis.append(f"Phase 2c metrics not loaded or empty: {metrics_csv}")
    if n_fail:
        diagnosis.append(
            "Some cells failed |Δ|<1e-6 join; check cell_id, embedding path, "
            "pred vs Phase 2c acc_target computation."
        )
    if n_pass == len(replay_rows) and replay_rows:
        diagnosis.append("All checked cells passed |Δ|<1e-6.")
    if do_replay:
        save_results_csv(readable / "replay_validation" / "replay_delta_table.csv", replay_rows)
        write_replay_validation_report(
            readable / "replay_validation" / "REPLAY_VALIDATION_REPORT.md",
            n_cells=len(replay_rows),
            n_pass=n_pass,
            n_fail=n_fail,
            max_abs_delta=max_abs,
            diagnosis=diagnosis or ["replay attempted"],
        )

    write_t3a_smoke_report(
        readable / "smoke" / "SMOKE_REPORT.md",
        status="passed" if all_rows and not any(
            r.get("method") == "load_failed" for r in all_rows
        ) else "failed_or_partial",
        selected_subjects=[str(s) for s in selected["subject"].tolist()],
        n_rows=len(all_rows),
        extra_notes=[
            "Round-1 scaffold smoke only; not a scientific conclusion.",
            f"t3a smoke config: {json.dumps(t3a_smoke)}" if run_t3a_flag else "t3a_minimal: skipped (run_t3a=false; no_tta only)",
            "Binary MI: entropy ranking ≡ max-confidence ranking (K=2).",
        ],
    )

    if oracle_rows:
        save_results_csv(readable / "oracle_diagnostic" / "oracle_results.csv", oracle_rows)
        write_oracle_diagnostic_report(
            readable / "oracle_diagnostic" / "ORACLE_DIAGNOSTIC_REPORT.md",
            status="minimal_oracle_ran",
            n_rows=len(oracle_rows),
            notes=[
                "Only target_label_oracle_proto implemented in Round-1.",
                PROVISIONAL_ORACLE_NOTES,
            ],
        )

    write_framework_smoke_report(
        readable / "reports" / "FRAMEWORK_SMOKE_REPORT.md",
        status="runnable",
        notes=[
            "TTA backend scaffold executed via phase3_tta runner.",
            f"mode={mode}, max_cells={max_cells}, n_result_rows={len(all_rows)}",
            "Pretrained model not integrated; use adapter/config later.",
            "Complex method-catalog candidates are not implemented in Round-1.",
        ],
    )

    summary.update(
        {
            "status": "ok",
            "n_cells": len(cell_specs),
            "n_result_rows": len(all_rows),
            "replay_n_pass": n_pass,
            "replay_n_fail": n_fail,
            "replay_max_abs_delta": max_abs,
            "selected_subjects": selected["subject"].tolist(),
            "readable_dir": str(readable),
            "smoke_csv": str(smoke_csv),
        }
    )
    (run_dir / "round1_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    logger.info("session_tta Round-1 done: %s", summary)
    return summary
