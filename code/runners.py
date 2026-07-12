#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""In-process experiment runners for the modular framework.

These functions replace the archived `scripts/*.py` entries. They import directly
from the `code.` packages (datasets / models / methods / experiments / training /
utils) so the new architecture is self-contained and runnable without restoring
the legacy `src/` + `scripts/` layers.

Each runner mirrors the behaviour of the original training script (same outputs,
same fairness/no-leakage guarantees) and is dispatched by `code/run.py`.

依赖: torch / numpy / scipy / sklearn / pandas / pyyaml.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional

CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parent


# --------------------------------------------------------------------------- #
# Small CLI helpers (shared)
# --------------------------------------------------------------------------- #
def _parse_list(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    vals = _parse_list(s)
    return None if vals is None else [int(v) for v in vals]


def _resolve_device(requested: str, logger):
    """cuda -> fail-fast if unavailable; auto -> cuda if available else cpu; cpu -> cpu."""
    import torch

    req = (requested or "auto").lower()
    avail = torch.cuda.is_available()
    if req == "cpu":
        return torch.device("cpu")
    if req == "cuda":
        if not avail:
            logger.error("device=cuda requested but CUDA is unavailable "
                         "(torch.version.cuda=%s). Run on a GPU node via srun/sbatch with "
                         "mi_torch_cu118, or pass --device cpu for a tiny CPU smoke test.",
                         torch.version.cuda)
            raise SystemExit(2)
        return torch.device("cuda")
    return torch.device("cuda" if avail else "cpu")


def _abs(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _resolve_manifest(cfg: Dict, P) -> Path:
    """Dataset-agnostic manifest resolution.

    优先用实验 config 里 `data.manifest`（绝对或相对仓库根路径）；缺省回退到
    paths.yaml 的全局 `processed_manifest`（WBCIC-SHU 默认）。这样同一套 runner
    可在 WBCIC-SHU / SHU 等不同数据集间切换，只改 config 不改代码。
    """
    m = (cfg.get("data") or {}).get("manifest")
    # 仅当 manifest 看起来是一个文件路径时才覆盖（WBCIC config 里 manifest 可能是
    # paths.yaml 的逻辑键名 "processed_manifest"，此时不覆盖，沿用 P.processed_manifest）。
    if m and ("/" in str(m) or str(m).endswith(".csv")):
        return _abs(str(m))
    return P.processed_manifest


def _build_spec(args: argparse.Namespace, train_cfg: Dict):
    from code.experiments.session_protocols import TrainSpec

    return TrainSpec(
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


def _cuda_info(device):
    import torch
    return {
        "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()), "device": str(device),
        "device_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
    }


def _write_rows(rows: List[Dict[str, object]], path: Path, columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})


# --------------------------------------------------------------------------- #
# Phase 0 — session drift diagnostic (CPU)
# --------------------------------------------------------------------------- #
def run_phase0_drift(cfg: Dict, cfg_path: Path, args: argparse.Namespace) -> None:
    from code.experiments.session_drift import (
        DriftParams, generate_figures, run_drift_analysis, summarize, write_markdown_report,
    )
    from code.datasets.session_splits import load_ok_sessions
    from code.utils.io import save_json
    from code.utils.logging_utils import get_logger
    from code.utils.paths import load_paths

    logger = get_logger("phase0_drift")
    t0 = time.time()
    P = load_paths(_abs(args.paths), require_raw=False)
    data_cfg = cfg.get("data", {})
    bands = cfg.get("bands", {})
    pcfg = cfg.get("params", {})
    subset = cfg.get("subset", {})

    params = DriftParams(
        fs=data_cfg.get("sfreq", 250),
        mu_band=tuple(bands.get("mu", [8, 13])),
        beta_band=tuple(bands.get("beta", [13, 30])),
        mmd_subsample=pcfg.get("mmd_subsample", 100),
        csp_components=pcfg.get("csp_components", 4),
        erd_baseline_ratio=pcfg.get("erd_baseline_ratio", 0.25),
        seed=pcfg.get("seed", 0),
    )
    subjects = _parse_list(args.subjects) or subset.get("subjects")
    max_subjects = args.max_subjects if args.max_subjects is not None else subset.get("max_subjects")

    out_dir = _abs(args.out) if args.out else _abs(
        cfg.get("output", {}).get("output_dir", "outputs/analysis/session_drift_v1"))
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_ok_sessions(
        _resolve_manifest(cfg, P), subjects=subjects,
        status_filter=tuple(data_cfg.get("status_filter", ["ok"])), max_subjects=max_subjects,
    )
    logger.info("loaded %d ok sessions / %d subjects", len(records), len({r.subject for r in records}))

    df = run_drift_analysis(records, params, logger=logger)
    df.to_csv(out_dir / "session_drift_report.csv", index=False, float_format="%.6f")
    summary = summarize(df, params)
    summary["run_id"] = cfg.get("run_id", "session_drift_v1")
    summary["n_sessions_loaded"] = len(records)
    save_json(summary, out_dir / "summary.json")
    figures = generate_figures(df, fig_dir, logger=logger)
    write_markdown_report(summary, figures, out_dir / "SESSION_DRIFT_REPORT.md",
                          report_run_id=summary["run_id"])
    logger.info("DONE in %.1fs | pairs=%d subjects=%d | %s",
                time.time() - t0, summary["n_pairs"], summary["n_subjects"], out_dir)


# --------------------------------------------------------------------------- #
# Phase 1 — within-session CV + single-source cross-session (GPU)
# --------------------------------------------------------------------------- #
def run_phase1_baseline(cfg: Dict, cfg_path: Path, args: argparse.Namespace) -> None:
    from code.datasets.session_splits import load_ok_sessions
    from code.experiments.session_protocols import (
        RESULT_COLUMNS, run_cross_session, run_within_session,
    )
    from code.utils.config import save_config
    from code.utils.io import save_json
    from code.utils.logging_utils import get_logger
    from code.utils.paths import load_paths

    logger = get_logger("phase1_baseline")
    t0 = time.time()
    P = load_paths(_abs(args.paths), require_raw=False)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    model_params_all = cfg.get("model_params", {}) or {}

    models = _parse_list(args.models) or cfg.get("models", ["eegnet"])
    protocols = (["within", "cross"] if (args.protocol or "both") == "both" else [args.protocol])
    protocols = [p for p in protocols if p in cfg.get("protocols", ["within", "cross"])] or protocols
    seeds = _parse_int_list(args.seeds) or train_cfg.get("seeds", [0])
    folds = args.folds if args.folds is not None else cfg.get("within", {}).get("folds", 10)

    spec = _build_spec(args, train_cfg)
    device = _resolve_device(args.device or train_cfg.get("device", "auto"), logger)
    data_dims = dict(
        n_channels=data_cfg.get("n_channels", 58), n_times=data_cfg.get("n_times", 1000),
        n_classes=data_cfg.get("n_classes", 2), sfreq=data_cfg.get("sfreq", 250),
    )

    out_dir = _abs(args.out) if args.out else _abs(
        cfg.get("output", {}).get("output_dir", "outputs/experiments/session_model_compare_v1"))
    ckpt_dir = None
    if not args.no_save_ckpt:
        ckpt_dir = _abs(cfg.get("output", {}).get("checkpoint_dir", "checkpoints/session_model_compare_v1"))
    runs_dir, splits_dir = out_dir / "runs", out_dir / "splits"
    runs_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "resolved_config.yaml")

    cuda_info = _cuda_info(device)
    logger.info("device=%s | cuda=%s", device, cuda_info)
    records = load_ok_sessions(
        _resolve_manifest(cfg, P), subjects=_parse_list(args.subjects), sessions=_parse_list(args.sessions),
        status_filter=tuple(data_cfg.get("status_filter", ["ok"])), max_subjects=args.max_subjects,
    )
    n_subj = len({r.subject for r in records})
    logger.info("loaded %d ok sessions / %d subjects | models=%s | protocols=%s | seeds=%s | folds=%d",
                len(records), n_subj, models, protocols, seeds, folds)

    for protocol in protocols:
        for model_name in models:
            mp = dict(model_params_all.get(model_name, {}))
            tag = f"{protocol}__{model_name}"
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
            seeds_in_rows = sorted({int(r["seed"]) for r in rows}) if rows else list(seeds)
            for sd in seeds_in_rows:
                srows = [r for r in rows if int(r["seed"]) == sd]
                _write_rows(srows, runs_dir / f"{tag}__seed{sd}.csv", RESULT_COLUMNS)
                save_json({"tag": tag, "protocol": protocol, "model": model_name,
                           "n_rows": len(srows), "spec": vars(spec), "data_dims": data_dims,
                           "cuda": cuda_info, "elapsed_sec": round(time.time() - t_run, 1)},
                          runs_dir / f"meta_{tag}__seed{sd}.json")
            logger.info("%s: %d rows (seeds=%s)", tag, len(rows), seeds_in_rows)
    logger.info("ALL DONE in %.1fs | outputs in %s", time.time() - t0, out_dir)


# --------------------------------------------------------------------------- #
# Phase 2a — multi-source cross-session ses-01+02 -> ses-03 (GPU)
# --------------------------------------------------------------------------- #
def run_phase2a_multisource(cfg: Dict, cfg_path: Path, args: argparse.Namespace) -> None:
    from code.datasets.session_splits import load_ok_sessions
    from code.experiments.session_multisource_protocols import (
        DEFAULT_TEST_SESSION, DEFAULT_TRAIN_SESSIONS, MULTISOURCE_RESULT_COLUMNS,
        run_multisource_cross_session,
    )
    from code.utils.config import save_config
    from code.utils.io import save_json
    from code.utils.logging_utils import get_logger
    from code.utils.paths import load_paths

    logger = get_logger("phase2a_multisource")
    t0 = time.time()
    P = load_paths(_abs(args.paths), require_raw=False)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    proto_cfg = cfg.get("multisource", {})
    train_sessions = proto_cfg.get("train_sessions", list(DEFAULT_TRAIN_SESSIONS))
    test_session = proto_cfg.get("test_session", DEFAULT_TEST_SESSION)
    models = _parse_list(args.models) or cfg.get("models", ["eegnet"])
    seeds = _parse_int_list(args.seeds) or train_cfg.get("seeds", [0])

    spec = _build_spec(args, train_cfg)
    device = _resolve_device(args.device or train_cfg.get("device", "auto"), logger)
    data_dims = dict(
        n_channels=data_cfg.get("n_channels", 58), n_times=data_cfg.get("n_times", 1000),
        n_classes=data_cfg.get("n_classes", 2), sfreq=data_cfg.get("sfreq", 250),
    )
    out_dir = _abs(args.out) if args.out else _abs(
        cfg.get("output", {}).get("output_dir", "outputs/experiments/session_multisource_v1"))
    runs_dir, splits_dir = out_dir / "runs", out_dir / "splits"
    runs_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "resolved_config.yaml")
    ckpt_dir = None
    if not args.no_save_ckpt:
        ckpt_dir = _abs(cfg.get("output", {}).get("checkpoint_dir", "checkpoints/session_multisource_v1"))

    cuda_info = _cuda_info(device)
    logger.info("device=%s | cuda=%s", device, cuda_info)
    records = load_ok_sessions(
        _resolve_manifest(cfg, P), subjects=_parse_list(args.subjects),
        status_filter=tuple(data_cfg.get("status_filter", ["ok"])), max_subjects=args.max_subjects,
    )
    logger.info("loaded %d ok sessions | models=%s | seeds=%s | %s -> %s",
                len(records), models, seeds, "+".join(train_sessions), test_session)

    model_params_all = cfg.get("model_params", {}) or {}
    for model_name in models:
        tag = f"multisource_0102_to_03__{model_name}"
        start = time.time()
        rows, used, skipped = run_multisource_cross_session(
            records, model_name=model_name, model_params=dict(model_params_all.get(model_name, {})),
            data_dims=data_dims, spec=spec, seeds=seeds, device=device,
            train_sessions=train_sessions, test_session=test_session,
            splits_dir=splits_dir, ckpt_dir=ckpt_dir, logger=logger,
        )
        seeds_in_rows = sorted({int(r["seed"]) for r in rows}) if rows else list(seeds)
        for seed in seeds_in_rows:
            seed_rows = [r for r in rows if int(r["seed"]) == seed]
            _write_rows(seed_rows, runs_dir / f"{tag}__seed{seed}.csv", MULTISOURCE_RESULT_COLUMNS)
            save_json({"tag": tag, "model": model_name, "train_sessions": train_sessions,
                       "test_session": test_session, "used_subjects": used,
                       "skipped_subjects": skipped, "spec": vars(spec), "data_dims": data_dims,
                       "cuda": cuda_info, "elapsed_sec": round(time.time() - start, 1)},
                      runs_dir / f"meta_{tag}__seed{seed}.json")
        logger.info("%s done: rows=%d used=%d skipped=%d", tag, len(rows), len(used), len(skipped))
    logger.info("ALL DONE in %.1fs | outputs in %s", time.time() - t0, out_dir)


# --------------------------------------------------------------------------- #
# Phase 2b — no-learning alignment baseline (GPU)
# --------------------------------------------------------------------------- #
def run_phase2b_alignment(cfg: Dict, cfg_path: Path, args: argparse.Namespace) -> None:
    from code.datasets.session_splits import load_ok_sessions
    from code.experiments.session_alignment_protocols import (
        ALIGNMENT_RESULT_COLUMNS, MULTISOURCE_TEST_SESSION, MULTISOURCE_TRAIN_SESSIONS,
        TRAINED_METHODS, enumerate_multisource_tasks, enumerate_single_source_tasks,
        run_alignment_tasks,
    )
    from code.utils.config import save_config
    from code.utils.io import save_json
    from code.utils.logging_utils import get_logger
    from code.utils.paths import load_paths

    logger = get_logger("phase2b_alignment")
    t0 = time.time()
    P = load_paths(_abs(args.paths), require_raw=False)
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

    spec = _build_spec(args, train_cfg)
    device = _resolve_device(args.device or train_cfg.get("device", "auto"), logger)
    sfreq = int(data_cfg.get("sfreq", 250))
    data_dims = dict(
        n_channels=data_cfg.get("n_channels", 58), n_times=data_cfg.get("n_times", 1000),
        n_classes=data_cfg.get("n_classes", 2), sfreq=sfreq,
    )
    fb = align_cfg.get("filterbank", {}) or {}
    align_params = {
        "eps": align_cfg.get("eps", 1e-5), "shrinkage": align_cfg.get("shrinkage", 0.1),
        "zscore_eps": align_cfg.get("zscore_eps", 1e-8), "bands": fb.get("bands"),
        "taps": fb.get("taps", 125), "w_min": fb.get("w_min", 0.5), "w_max": fb.get("w_max", 2.0),
    }

    out_dir = _abs(args.out) if args.out else _abs(
        cfg.get("output", {}).get("output_dir", "outputs/experiments/alignment_baseline_v1"))
    cross_dir = out_dir / "cross_session"
    runs_dir, splits_dir = cross_dir / "runs", cross_dir / "splits"
    runs_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "resolved_config_summary.yaml")
    ckpt_dir = None
    if not args.no_save_ckpt:
        ckpt_dir = _abs(args.ckpt_dir or cfg.get("output", {}).get(
            "checkpoint_dir", "checkpoints/alignment_baseline_v1"))

    cuda_info = _cuda_info(device)
    logger.info("device=%s | cuda=%s", device, cuda_info)
    records = load_ok_sessions(
        _resolve_manifest(cfg, P), subjects=_parse_list(args.subjects),
        status_filter=tuple(data_cfg.get("status_filter", ["ok"])), max_subjects=args.max_subjects,
    )

    tasks: List = []
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
        tasks, methods=methods, models=models, model_params_all=cfg.get("model_params", {}) or {},
        data_dims=data_dims, spec=spec, seeds=seeds, device=device, sfreq=sfreq,
        align_params=align_params, splits_dir=splits_dir, ckpt_dir=ckpt_dir,
        save_ckpt=not args.no_save_ckpt, logger=logger,
    )

    suffix = f"__{args.tag_suffix}" if args.tag_suffix else ""
    groups: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        key = f"{r['method']}__{r['model']}__{r['training_scope']}__seed{r['seed']}{suffix}"
        groups.setdefault(key, []).append(r)
    for key, grp in groups.items():
        _write_rows(grp, runs_dir / f"alignment__{key}.csv", ALIGNMENT_RESULT_COLUMNS)
    save_json({"experiment_id": "alignment_baseline_v1", "methods": methods, "models": models,
               "seeds": seeds, "protocol_groups": proto_groups, "n_tasks": len(tasks),
               "n_rows": len(rows), "n_ok": sum(1 for r in rows if r["status"] == "ok"),
               "n_failed": sum(1 for r in rows if r["status"] == "failed"),
               "multisource_skipped": ms_skipped, "spec": vars(spec), "align_params": align_params,
               "data_dims": data_dims, "cuda": cuda_info, "elapsed_sec": round(time.time() - t0, 1)},
              runs_dir / f"meta_alignment{suffix}.json")
    logger.info("ALL DONE in %.1fs | rows=%d ok=%d failed=%d | out=%s", time.time() - t0, len(rows),
                sum(1 for r in rows if r["status"] == "ok"),
                sum(1 for r in rows if r["status"] == "failed"), out_dir)


# --------------------------------------------------------------------------- #
# Phase 2c — prototype drift analysis (frozen-model diagnostic, GPU)
# --------------------------------------------------------------------------- #
def run_phase2c_prototype_drift(cfg: Dict, cfg_path: Path, args: argparse.Namespace) -> None:
    from code.datasets.session_splits import load_ok_sessions
    from code.experiments.prototype_drift import (
        INDEX_COLUMNS, METRIC_COLUMNS, PROTOTYPE_COLUMNS, STATUS_COLUMNS,
        run_prototype_drift,
    )
    from code.utils.config import save_config
    from code.utils.io import save_json
    from code.utils.logging_utils import get_logger
    from code.utils.paths import load_paths

    logger = get_logger("phase2c_prototype_drift")
    t0 = time.time()
    P = load_paths(_abs(args.paths), require_raw=False)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    run_id = cfg.get("run_id", "prototype_drift_v1")
    dataset = cfg.get("dataset", "wbci_shu")
    task = cfg.get("task", "2C left vs right MI")
    prototype_types = cfg.get("prototype_types", ["label_based", "confidence_weighted", "correct_only"])
    distances = cfg.get("distances", ["euclidean", "cosine"])

    models = _parse_list(args.models) or cfg.get("models", ["eegnet"])
    seeds = _parse_int_list(args.seeds) or cfg.get("seeds") or train_cfg.get("seeds", [0])

    spec = _build_spec(args, train_cfg)
    device = _resolve_device(args.device or train_cfg.get("device", "auto"), logger)
    data_dims = dict(
        n_channels=data_cfg.get("n_channels", 58), n_times=data_cfg.get("n_times", 1000),
        n_classes=data_cfg.get("n_classes", 2), sfreq=data_cfg.get("sfreq", 250),
    )

    out_cfg = cfg.get("output", {}) or {}
    out_dir = _abs(args.out) if args.out else _abs(
        out_cfg.get("run_dir", out_cfg.get("output_dir", "outputs/experiments/prototype_drift_v1")))
    runs_dir = out_dir / "runs"
    # Embeddings always nest under the (possibly overridden) run dir so a --out smoke
    # run can never write into the full-run embeddings tree.
    embed_dir = out_dir / "embeddings"
    runs_dir.mkdir(parents=True, exist_ok=True)
    embed_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "resolved_config.yaml")
    ckpt_dir = None
    if not args.no_save_ckpt:
        ckpt_dir = _abs(args.ckpt_dir or out_cfg.get("checkpoint_dir", "checkpoints/prototype_drift_v1"))

    cuda_info = _cuda_info(device)
    logger.info("device=%s | cuda=%s", device, cuda_info)
    records = load_ok_sessions(
        _resolve_manifest(cfg, P), subjects=_parse_list(args.subjects), sessions=_parse_list(args.sessions),
        status_filter=tuple(data_cfg.get("status_filter", ["ok"])), max_subjects=args.max_subjects,
    )
    n_subj = len({r.subject for r in records})
    logger.info("loaded %d ok sessions / %d subjects | models=%s | seeds=%s | ptypes=%s | dists=%s",
                len(records), n_subj, models, seeds, prototype_types, distances)

    for model_name in models:
        mp = dict((cfg.get("model_params", {}) or {}).get(model_name, {}))
        for seed in seeds:
            start = time.time()
            res = run_prototype_drift(
                records, model_name=model_name, model_params=mp, data_dims=data_dims,
                spec=spec, seed=int(seed), device=device, prototype_types=prototype_types,
                distances=distances, run_id=run_id, dataset=dataset, task=task,
                embed_dir=embed_dir, ckpt_dir=ckpt_dir, logger=logger,
            )
            tag = f"{model_name}__seed{seed}"
            _write_rows(res["metric_rows"], runs_dir / f"metrics__{tag}.csv", METRIC_COLUMNS)
            _write_rows(res["prototype_rows"], runs_dir / f"prototypes__{tag}.csv", PROTOTYPE_COLUMNS)
            _write_rows(res["index_rows"], runs_dir / f"embed_index__{tag}.csv", INDEX_COLUMNS)
            _write_rows(res["status_rows"], runs_dir / f"status__{tag}.csv", STATUS_COLUMNS)
            n_ok = sum(1 for s in res["status_rows"] if s["status"] == "ok")
            n_failed = sum(1 for s in res["status_rows"] if s["status"] == "failed")
            save_json({"run_id": run_id, "model": model_name, "seed": int(seed),
                       "n_metric_rows": len(res["metric_rows"]),
                       "n_cells_ok": n_ok, "n_cells_failed": n_failed,
                       "prototype_types": prototype_types, "distances": distances,
                       "spec": vars(spec), "data_dims": data_dims, "cuda": cuda_info,
                       "elapsed_sec": round(time.time() - start, 1)},
                      runs_dir / f"meta__{tag}.json")
            logger.info("%s done: metric_rows=%d cells_ok=%d failed=%d (%.1fs)",
                        tag, len(res["metric_rows"]), n_ok, n_failed, time.time() - start)
    logger.info("ALL DONE in %.1fs | outputs in %s", time.time() - t0, out_dir)


def run_phase3_tta(cfg: Dict, cfg_path: Path, args: argparse.Namespace) -> None:
    """Phase 3 Round-1: model-agnostic TTA backend scaffold + minimal smoke.

    Defaults to tiny CPU smoke (no full sweep). Heavy GPU / full ablation are
    out of Round-1 scope.
    """
    from code.experiments.session_tta import run_session_tta
    from code.utils.logging_utils import get_logger

    logger = get_logger("phase3_tta")
    # CLI safety overrides into round1 block
    round1 = dict(cfg.get("round1") or {})
    if getattr(args, "dry_run", False):
        round1["dry_run"] = True
    if getattr(args, "subjects", None):
        # subjects CLI is informational for Round-1; auto-select remains default
        round1["cli_subjects"] = args.subjects
    cfg = dict(cfg)
    cfg["round1"] = round1
    logger.info(
        "phase3_tta Round-1 starting (mode=%s max_cells=%s)",
        round1.get("mode", "smoke"),
        round1.get("max_cells", 4),
    )
    summary = run_session_tta(cfg, project_root=PROJECT_ROOT)
    logger.info("phase3_tta finished: status=%s rows=%s",
                summary.get("status"), summary.get("n_result_rows"))


PHASE_RUNNERS = {
    "phase0_drift_diagnostic": run_phase0_drift,
    "phase1_baseline": run_phase1_baseline,
    "phase2a_multisource": run_phase2a_multisource,
    "phase2b_alignment": run_phase2b_alignment,
    "phase2c_prototype_drift": run_phase2c_prototype_drift,
    "phase3_tta": run_phase3_tta,
}
