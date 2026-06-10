#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train Step-2 no-learning alignment baselines (single-source + multi-source).

Fairness: model trains on the SOURCE session(s); the target session is used only
through its unlabeled X for alignment statistics; y_test only for final eval.
See configs/session_alignment_compare.yaml and the protocol docstring.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from src.data.session_splits import load_ok_sessions  # noqa: E402
from src.evaluation.session_alignment_protocols import (  # noqa: E402
    ALIGNMENT_RESULT_COLUMNS,
    MULTISOURCE_TEST_SESSION,
    MULTISOURCE_TRAIN_SESSIONS,
    TRAINED_METHODS,
    enumerate_multisource_tasks,
    enumerate_single_source_tasks,
    run_alignment_tasks,
)
from src.evaluation.session_protocols import TrainSpec  # noqa: E402
from src.utils.config import load_config, save_config  # noqa: E402
from src.utils.io import save_json  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.utils.paths import load_paths  # noqa: E402

logger = get_logger("train_session_alignment")


def _parse_list(s: Optional[str]):
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_int_list(s: Optional[str]):
    vals = _parse_list(s)
    return None if vals is None else [int(v) for v in vals]


def _resolve_device(requested: str) -> torch.device:
    req = (requested or "auto").lower()
    if req == "cpu":
        return torch.device("cpu")
    if req == "cuda":
        if not torch.cuda.is_available():
            logger.error("device=cuda requested but CUDA is unavailable. Not falling back to CPU.")
            raise SystemExit(2)
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _write_rows(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ALIGNMENT_RESULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in ALIGNMENT_RESULT_COLUMNS})


def main() -> None:
    ap = argparse.ArgumentParser(description="Step-2 alignment baseline trainer.")
    ap.add_argument("--config", default="configs/session_alignment_compare.yaml")
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--methods", default=None, help="comma list; default = config methods")
    ap.add_argument("--models", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--protocols", default=None, help="single,multi (default both)")
    ap.add_argument("--subjects", default=None)
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--val-fraction", type=float, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt-dir", default=None, help="override checkpoint dir (e.g. for smoke tests)")
    ap.add_argument("--tag-suffix", default="", help="appended to run CSV filenames (avoid clobber)")
    ap.add_argument("--no-save-ckpt", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    paths = load_paths(PROJECT_ROOT / args.paths, require_raw=False)
    cfg = load_config(PROJECT_ROOT / args.config)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    align_cfg = cfg.get("alignment", {}) or {}
    ms_cfg = cfg.get("multisource", {}) or {}
    ms_train = ms_cfg.get("train_sessions", list(MULTISOURCE_TRAIN_SESSIONS))
    ms_test = ms_cfg.get("test_session", MULTISOURCE_TEST_SESSION)

    methods = _parse_list(args.methods) or cfg.get("methods", list(TRAINED_METHODS))
    methods = [m for m in methods if m in TRAINED_METHODS]
    if not methods:
        logger.error("no trainable methods selected (none_reference is reference-only).")
        raise SystemExit(2)
    models = _parse_list(args.models) or cfg.get("models", ["eegnet"])
    seeds = _parse_int_list(args.seeds) or train_cfg.get("seeds", [0])
    proto_groups = _parse_list(args.protocols) or cfg.get("protocols", ["single", "multi"])

    spec = TrainSpec(
        batch_size=args.batch_size or train_cfg.get("batch_size", 16),
        lr=args.lr or train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        optimizer=train_cfg.get("optimizer", "adam"),
        max_epochs=args.max_epochs or train_cfg.get("max_epochs", 100),
        early_stopping_patience=(args.patience if args.patience is not None
                                 else train_cfg.get("early_stopping_patience", 20)),
        val_fraction=(args.val_fraction if args.val_fraction is not None
                      else train_cfg.get("val_fraction", 0.2)),
        num_workers=(args.num_workers if args.num_workers is not None
                     else train_cfg.get("num_workers", 2)),
    )
    device = _resolve_device(args.device or train_cfg.get("device", "auto"))
    sfreq = int(data_cfg.get("sfreq", 250))
    data_dims = {
        "n_channels": data_cfg.get("n_channels", 58),
        "n_times": data_cfg.get("n_times", 1000),
        "n_classes": data_cfg.get("n_classes", 2),
        "sfreq": sfreq,
    }
    align_params = {
        "eps": align_cfg.get("eps", 1e-5),
        "shrinkage": align_cfg.get("shrinkage", 0.1),
        "zscore_eps": align_cfg.get("zscore_eps", 1e-8),
        "bands": (align_cfg.get("filterbank", {}) or {}).get("bands"),
        "taps": (align_cfg.get("filterbank", {}) or {}).get("taps", 125),
        "w_min": (align_cfg.get("filterbank", {}) or {}).get("w_min", 0.5),
        "w_max": (align_cfg.get("filterbank", {}) or {}).get("w_max", 2.0),
    }

    out_dir = Path(args.out) if args.out else PROJECT_ROOT / cfg.get(
        "output", {}).get("output_dir", "outputs/experiments/alignment_baseline_v1")
    out_dir = out_dir if out_dir.is_absolute() else PROJECT_ROOT / out_dir
    cross_dir = out_dir / "cross_session"
    runs_dir = cross_dir / "runs"
    splits_dir = cross_dir / "splits"
    runs_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "resolved_config_summary.yaml")

    ckpt_dir = None
    if not args.no_save_ckpt:
        raw_ckpt = args.ckpt_dir or cfg.get("output", {}).get("checkpoint_dir",
                                                              "checkpoints/alignment_baseline_v1")
        ckpt_dir = Path(raw_ckpt)
        ckpt_dir = ckpt_dir if ckpt_dir.is_absolute() else PROJECT_ROOT / ckpt_dir

    cuda_info = {
        "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()), "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    logger.info("device=%s | cuda=%s", device, cuda_info)

    records = load_ok_sessions(
        paths.processed_manifest,
        subjects=_parse_list(args.subjects),
        status_filter=tuple(data_cfg.get("status_filter", ["ok"])),
        max_subjects=args.max_subjects,
    )

    tasks = []
    ms_skipped: List[Dict[str, str]] = []
    if "single" in proto_groups:
        tasks += enumerate_single_source_tasks(records)
    if "multi" in proto_groups:
        ms_tasks, ms_skipped = enumerate_multisource_tasks(
            records, train_sessions=ms_train, test_session=ms_test)
        tasks += ms_tasks
    logger.info("loaded %d ok sessions | methods=%s | models=%s | seeds=%s | groups=%s | tasks=%d",
                len(records), methods, models, seeds, proto_groups, len(tasks))

    rows = run_alignment_tasks(
        tasks, methods=methods, models=models,
        model_params_all=cfg.get("model_params", {}) or {}, data_dims=data_dims,
        spec=spec, seeds=seeds, device=device, sfreq=sfreq, align_params=align_params,
        splits_dir=splits_dir, ckpt_dir=ckpt_dir, save_ckpt=not args.no_save_ckpt, logger=logger,
    )

    suffix = f"__{args.tag_suffix}" if args.tag_suffix else ""
    # One CSV per (method, model, training_scope, seed) to keep parallel jobs collision-free.
    groups: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        key = f"{r['method']}__{r['model']}__{r['training_scope']}__seed{r['seed']}{suffix}"
        groups.setdefault(key, []).append(r)
    for key, grp in groups.items():
        _write_rows(grp, runs_dir / f"alignment__{key}.csv")

    save_json({
        "experiment_id": "alignment_baseline_v1",
        "methods": methods, "models": models, "seeds": seeds,
        "protocol_groups": proto_groups, "n_tasks": len(tasks),
        "n_rows": len(rows), "n_ok": sum(1 for r in rows if r["status"] == "ok"),
        "n_failed": sum(1 for r in rows if r["status"] == "failed"),
        "multisource_skipped": ms_skipped, "spec": vars(spec),
        "align_params": align_params, "data_dims": data_dims, "cuda": cuda_info,
        "elapsed_sec": round(time.time() - t0, 1),
    }, runs_dir / f"meta_alignment{('__' + args.tag_suffix) if args.tag_suffix else ''}.json")

    logger.info("ALL DONE in %.1fs | rows=%d ok=%d failed=%d | out=%s",
                time.time() - t0, len(rows),
                sum(1 for r in rows if r["status"] == "ok"),
                sum(1 for r in rows if r["status"] == "failed"), out_dir)


if __name__ == "__main__":
    main()
