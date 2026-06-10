#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified within-session CV / cross-session training entry for the 4 comparison models.

Runs EEGNet / DeepConvNet / FBCNet / CAP-EEGNet under ONE protocol + data filter +
metric set so baselines and our model are compared fairly and leakage-free. Data
entry = status=ok per-session .npz from processed_manifest.csv (configs/paths.yaml);
never reads derivatives .mat or raw BDF, never writes raw/workspace2.

Examples (smoke tests — run on a GPU compute node via srun, NOT the login node):

  python scripts/train_session_models.py --config configs/session_model_compare.yaml \
      --models eegnet,deepconvnet,fbcnet,cap_eegnet --protocol within \
      --subjects 1,2 --folds 2 --max-epochs 3 --device cuda

  python scripts/train_session_models.py --config configs/session_model_compare.yaml \
      --models eegnet,deepconvnet,fbcnet,cap_eegnet --protocol cross \
      --subjects 1,2 --max-epochs 3 --device cuda

Outputs (under output_dir from the config, default outputs/experiments/session_model_compare_v1):
  runs/{protocol}__{model}.csv   one row per (session,fold,seed) or (pair,seed)
  runs/meta_{protocol}__{model}.json  timing / device / args
  splits/*.json                  exact fold indices / directed pairs (reproducible)
  resolved_config.yaml           the config actually used
Checkpoints go under checkpoint_dir (default checkpoints/session_model_compare_v1).
Aggregate tables/figures/report are produced separately by scripts/summarize_session_results.py.
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
from src.evaluation.session_protocols import (  # noqa: E402
    RESULT_COLUMNS,
    TrainSpec,
    run_cross_session,
    run_within_session,
)
from src.utils.config import load_config, save_config  # noqa: E402
from src.utils.io import save_json  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.utils.paths import load_paths  # noqa: E402

logger = get_logger("train_session_models")


def _parse_list(s: Optional[str]):
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_int_list(s: Optional[str]):
    vals = _parse_list(s)
    return None if vals is None else [int(v) for v in vals]


def _resolve_device(requested: str) -> torch.device:
    """cuda -> fail-fast if unavailable; auto -> cuda if available else cpu; cpu -> cpu."""
    req = (requested or "auto").lower()
    avail = torch.cuda.is_available()
    if req == "cpu":
        return torch.device("cpu")
    if req == "cuda":
        if not avail:
            logger.error("device=cuda requested but CUDA is unavailable "
                         "(torch.version.cuda=%s). Run on a GPU node via srun/sbatch with the "
                         "CUDA-enabled env (e.g. mi_torch_cu118), or pass --device cpu for a tiny "
                         "CPU smoke test.", torch.version.cuda)
            raise SystemExit(2)
        return torch.device("cuda")
    return torch.device("cuda" if avail else "cpu")


def _write_rows(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in RESULT_COLUMNS})


def main() -> None:
    ap = argparse.ArgumentParser(description="Within/Cross-session model comparison trainer.")
    ap.add_argument("--config", default="configs/session_model_compare.yaml")
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--models", default=None, help="comma list; default = config models")
    ap.add_argument("--protocol", default="both", choices=["within", "cross", "both"])
    ap.add_argument("--subjects", default=None, help="comma list, e.g. 1,2")
    ap.add_argument("--sessions", default=None, help="comma list, e.g. 1,2,3")
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--folds", type=int, default=None, help="within-session K (override config)")
    ap.add_argument("--seeds", default=None, help="comma list (override config train.seeds)")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--val-fraction", type=float, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--device", default=None, help="cuda | cpu | auto (override config)")
    ap.add_argument("--out", default=None, help="override output_dir")
    ap.add_argument("--no-save-ckpt", action="store_true", help="do not write checkpoints")
    args = ap.parse_args()

    t0 = time.time()
    P = load_paths(PROJECT_ROOT / args.paths, require_raw=False)
    cfg = load_config(PROJECT_ROOT / args.config)

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    model_params_all = cfg.get("model_params", {}) or {}

    models = _parse_list(args.models) or cfg.get("models", ["eegnet"])
    protocols = (["within", "cross"] if args.protocol == "both"
                 else [args.protocol])
    # keep only protocols also enabled in config (defensive)
    protocols = [p for p in protocols if p in cfg.get("protocols", ["within", "cross"])] or protocols

    seeds = _parse_int_list(args.seeds) or train_cfg.get("seeds", [0])
    folds = args.folds if args.folds is not None else cfg.get("within", {}).get("folds", 10)

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
    data_dims = dict(
        n_channels=data_cfg.get("n_channels", 58),
        n_times=data_cfg.get("n_times", 1000),
        n_classes=data_cfg.get("n_classes", 2),
        sfreq=data_cfg.get("sfreq", 250),
    )

    out_dir = Path(args.out) if args.out else (PROJECT_ROOT / cfg.get("output", {}).get(
        "output_dir", "outputs/experiments/session_model_compare_v1"))
    out_dir = out_dir if out_dir.is_absolute() else (PROJECT_ROOT / out_dir)
    ckpt_dir = None
    if not args.no_save_ckpt:
        ck = cfg.get("output", {}).get("checkpoint_dir", "checkpoints/session_model_compare_v1")
        ckpt_dir = Path(ck) if Path(ck).is_absolute() else (PROJECT_ROOT / ck)
    runs_dir = out_dir / "runs"
    splits_dir = out_dir / "splits"
    runs_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "resolved_config.yaml")

    cuda_info = {
        "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "device_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
    }
    logger.info("device=%s | cuda=%s", device, cuda_info)

    subjects = _parse_list(args.subjects)
    sessions = _parse_list(args.sessions)
    records = load_ok_sessions(
        P.processed_manifest, subjects=subjects, sessions=sessions,
        status_filter=tuple(data_cfg.get("status_filter", ["ok"])),
        max_subjects=args.max_subjects,
    )
    n_subj = len({r.subject for r in records})
    logger.info("loaded %d ok sessions across %d subjects | models=%s | protocols=%s | seeds=%s | folds=%d",
                len(records), n_subj, models, protocols, seeds, folds)

    for protocol in protocols:
        for model_name in models:
            mp = dict(model_params_all.get(model_name, {}))
            tag = f"{protocol}__{model_name}"
            logger.info("=== RUN %s ===", tag)
            t_run = time.time()
            if protocol == "within":
                rows = run_within_session(
                    records, model_name=model_name, model_params=mp, data_dims=data_dims,
                    spec=spec, folds=folds, seeds=seeds, device=device,
                    splits_dir=splits_dir, ckpt_dir=ckpt_dir, logger=logger,
                )
            else:
                rows = run_cross_session(
                    records, model_name=model_name, model_params=mp, data_dims=data_dims,
                    spec=spec, seeds=seeds, device=device,
                    splits_dir=splits_dir, ckpt_dir=ckpt_dir, logger=logger,
                )
            # Write ONE CSV per (tag, seed) so different seed jobs never overwrite each
            # other: runs/{protocol}__{model}__seed{seed}.csv. The summarizer globs all
            # CSVs and reads the `seed` column, so mixed naming is fine.
            seeds_in_rows = sorted({int(r["seed"]) for r in rows}) if rows else list(seeds)
            for sd in seeds_in_rows:
                srows = [r for r in rows if int(r["seed"]) == sd]
                _write_rows(srows, runs_dir / f"{tag}__seed{sd}.csv")
            meta = {
                "tag": tag, "protocol": protocol, "model": model_name,
                "n_rows": len(rows), "n_sessions": len(records), "n_subjects": n_subj,
                "models": models, "seeds": seeds, "folds": folds,
                "spec": vars(spec), "data_dims": data_dims, "model_params": mp,
                "subjects_arg": subjects, "max_subjects": args.max_subjects,
                "cuda": cuda_info, "elapsed_sec": round(time.time() - t_run, 1),
            }
            for sd in seeds_in_rows:
                save_json(meta, runs_dir / f"meta_{tag}__seed{sd}.json")
            logger.info("%s: %d rows -> %s (seeds=%s, %.1fs)", tag, len(rows),
                        runs_dir / f"{tag}__seed*.csv", seeds_in_rows, meta["elapsed_sec"])

    logger.info("ALL DONE in %.1fs | outputs in %s", time.time() - t0, out_dir)


if __name__ == "__main__":
    main()
