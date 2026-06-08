#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run the cross-session drift diagnostic (direction A) and write all outputs.

CPU-only (numpy/scipy/sklearn; no torch/GPU). Data entry = status=ok per-session
.npz from processed_manifest.csv via configs/paths.yaml; never reads derivatives
.mat or raw BDF; writes only under the configured output_dir (outputs/).

Examples (run on a compute node via srun; heavy jobs never on the login node):
  # smoke test on 2 subjects
  python scripts/analysis/run_session_drift.py --config configs/session_drift.yaml --subjects 1,2
  # full (all ok subjects)
  python scripts/analysis/run_session_drift.py --config configs/session_drift.yaml
  # cap the number of subjects
  python scripts/analysis/run_session_drift.py --config configs/session_drift.yaml --max-subjects 10

Outputs (under output_dir, default outputs/analysis/session_drift_v1):
  session_drift_report.csv   one row per within-subject session pair
  summary.json               mean/median/std per metric + counts
  SESSION_DRIFT_REPORT.md    human-readable summary + figure index
  figures/*.png              matplotlib figures
See docs/SESSION_DRIFT_ANALYSIS.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.session_drift import (  # noqa: E402
    DriftParams,
    generate_figures,
    run_drift_analysis,
    summarize,
    write_markdown_report,
)
from src.data.session_splits import load_ok_sessions  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.io import save_json  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.utils.paths import load_paths  # noqa: E402

logger = get_logger("run_session_drift")


def _parse_list(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-session drift diagnostic (direction A).")
    ap.add_argument("--config", default="configs/session_drift.yaml")
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--subjects", default=None, help="comma list, e.g. 1,2 (smoke test)")
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--out", default=None, help="override output_dir")
    args = ap.parse_args()

    t0 = time.time()
    P = load_paths(PROJECT_ROOT / args.paths, require_raw=False)
    cfg = load_config(PROJECT_ROOT / args.config)
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

    out_dir = Path(args.out) if args.out else Path(cfg.get("output", {}).get(
        "output_dir", "outputs/analysis/session_drift_v1"))
    out_dir = out_dir if out_dir.is_absolute() else (PROJECT_ROOT / out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_ok_sessions(
        P.processed_manifest, subjects=subjects,
        status_filter=tuple(data_cfg.get("status_filter", ["ok"])),
        max_subjects=max_subjects,
    )
    n_subj = len({r.subject for r in records})
    logger.info("loaded %d ok sessions / %d subjects (subjects=%s, max_subjects=%s)",
                len(records), n_subj, subjects, max_subjects)

    df = run_drift_analysis(records, params, logger=logger)
    csv_path = out_dir / "session_drift_report.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    logger.info("wrote %s (%d rows)", csv_path, len(df))

    summary = summarize(df, params)
    summary["run_id"] = cfg.get("run_id", "session_drift_v1")
    summary["n_sessions_loaded"] = len(records)
    save_json(summary, out_dir / "summary.json")

    figures = generate_figures(df, fig_dir, logger=logger)
    write_markdown_report(summary, figures, out_dir / "SESSION_DRIFT_REPORT.md",
                          report_run_id=summary["run_id"])

    logger.info("DONE in %.1fs | pairs=%d subjects=%d | outputs in %s",
                time.time() - t0, summary["n_pairs"], summary["n_subjects"], out_dir)


if __name__ == "__main__":
    main()
