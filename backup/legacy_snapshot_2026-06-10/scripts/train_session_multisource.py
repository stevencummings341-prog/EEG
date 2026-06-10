#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train multi-source cross-session baseline: ses-01 + ses-02 -> ses-03."""

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
from src.evaluation.session_multisource_protocols import (  # noqa: E402
    DEFAULT_TEST_SESSION,
    DEFAULT_TRAIN_SESSIONS,
    MULTISOURCE_RESULT_COLUMNS,
    run_multisource_cross_session,
)
from src.evaluation.session_protocols import TrainSpec  # noqa: E402
from src.utils.config import load_config, save_config  # noqa: E402
from src.utils.io import save_json  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.utils.paths import load_paths  # noqa: E402

logger = get_logger("train_session_multisource")


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
        writer = csv.DictWriter(f, fieldnames=MULTISOURCE_RESULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in MULTISOURCE_RESULT_COLUMNS})


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-source cross-session trainer.")
    ap.add_argument("--config", default="configs/session_multisource_compare.yaml")
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--models", default=None)
    ap.add_argument("--subjects", default=None)
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--val-fraction", type=float, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-save-ckpt", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    paths = load_paths(PROJECT_ROOT / args.paths, require_raw=False)
    cfg = load_config(PROJECT_ROOT / args.config)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    proto_cfg = cfg.get("multisource", {})
    train_sessions = proto_cfg.get("train_sessions", list(DEFAULT_TRAIN_SESSIONS))
    test_session = proto_cfg.get("test_session", DEFAULT_TEST_SESSION)
    models = _parse_list(args.models) or cfg.get("models", ["eegnet"])
    seeds = _parse_int_list(args.seeds) or train_cfg.get("seeds", [0])

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
    data_dims = {
        "n_channels": data_cfg.get("n_channels", 58),
        "n_times": data_cfg.get("n_times", 1000),
        "n_classes": data_cfg.get("n_classes", 2),
        "sfreq": data_cfg.get("sfreq", 250),
    }

    out_dir = Path(args.out) if args.out else PROJECT_ROOT / cfg.get(
        "output", {}).get("output_dir", "outputs/experiments/session_multisource_v1")
    out_dir = out_dir if out_dir.is_absolute() else PROJECT_ROOT / out_dir
    runs_dir = out_dir / "runs"
    splits_dir = out_dir / "splits"
    runs_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "resolved_config.yaml")

    ckpt_dir = None
    if not args.no_save_ckpt:
        raw_ckpt = cfg.get("output", {}).get("checkpoint_dir", "checkpoints/session_multisource_v1")
        ckpt_dir = Path(raw_ckpt)
        ckpt_dir = ckpt_dir if ckpt_dir.is_absolute() else PROJECT_ROOT / ckpt_dir

    cuda_info = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    logger.info("device=%s | cuda=%s", device, cuda_info)

    records = load_ok_sessions(
        paths.processed_manifest,
        subjects=_parse_list(args.subjects),
        status_filter=tuple(data_cfg.get("status_filter", ["ok"])),
        max_subjects=args.max_subjects,
    )
    logger.info("loaded %d ok sessions | models=%s | seeds=%s | %s -> %s",
                len(records), models, seeds, "+".join(train_sessions), test_session)

    model_params_all = cfg.get("model_params", {}) or {}
    for model_name in models:
        tag = f"multisource_0102_to_03__{model_name}"
        start = time.time()
        rows, used, skipped = run_multisource_cross_session(
            records,
            model_name=model_name,
            model_params=dict(model_params_all.get(model_name, {})),
            data_dims=data_dims,
            spec=spec,
            seeds=seeds,
            device=device,
            train_sessions=train_sessions,
            test_session=test_session,
            splits_dir=splits_dir,
            ckpt_dir=ckpt_dir,
            logger=logger,
        )
        seeds_in_rows = sorted({int(r["seed"]) for r in rows}) if rows else list(seeds)
        for seed in seeds_in_rows:
            seed_rows = [r for r in rows if int(r["seed"]) == seed]
            _write_rows(seed_rows, runs_dir / f"{tag}__seed{seed}.csv")
            save_json({
                "tag": tag,
                "protocol": "multisource_0102_to_03",
                "model": model_name,
                "train_sessions": train_sessions,
                "test_session": test_session,
                "used_subjects": used,
                "skipped_subjects": skipped,
                "seeds": seeds,
                "spec": vars(spec),
                "data_dims": data_dims,
                "cuda": cuda_info,
                "elapsed_sec": round(time.time() - start, 1),
            }, runs_dir / f"meta_{tag}__seed{seed}.json")
        logger.info("%s done: rows=%d used=%d skipped=%d elapsed=%.1fs",
                    tag, len(rows), len(used), len(skipped), time.time() - start)

    logger.info("ALL DONE in %.1fs | outputs in %s", time.time() - t0, out_dir)


if __name__ == "__main__":
    main()
