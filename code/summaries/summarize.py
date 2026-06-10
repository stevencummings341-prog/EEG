#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize dispatcher: run the matching phase summarizer + canonical report.

Called by `code/run.py --summarize`. Reads the experiment's `output_dir` from the
phase config, invokes the native summarizer (tables/figures/native report), then
writes the canonical 9-section `REPORT.md` (see code/summaries/canonical.py).

依赖: pandas, numpy, matplotlib.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from . import alignment as _alignment
from . import canonical as _canonical
from . import multisource as _multisource
from . import session as _session

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _abs(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def summarize_phase(phase: str, cfg: Dict) -> None:
    out_cfg = cfg.get("output", {}) or {}
    if phase == "phase1_baseline":
        out_dir = _abs(out_cfg.get("output_dir", "outputs/experiments/session_model_compare_v1"))
        _session.main(["--results-dir", str(out_dir)])
        report = _canonical.write_canonical_report(phase, out_dir)
    elif phase == "phase2a_multisource":
        out_dir = _abs(out_cfg.get("output_dir", "outputs/experiments/session_multisource_v1"))
        base = _abs("outputs/experiments/session_model_compare_v1/summaries")
        _multisource.main(["--out", str(out_dir), "--baseline-summaries", str(base)])
        report = _canonical.write_canonical_report(phase, out_dir)
    elif phase == "phase2b_alignment":
        out_dir = _abs(out_cfg.get("output_dir", "outputs/experiments/alignment_baseline_v1"))
        base = out_cfg.get("baseline_cross_all",
                           "outputs/experiments/baseline_v1/cross_session/tables/results_cross_session_all.csv")
        drift = out_cfg.get("drift_per_subject",
                            "outputs/analysis/session_drift_v1/per_subject_drift_summary.csv")
        _alignment.main(["--out", str(out_dir), "--baseline-cross-all", str(_abs(base)),
                         "--drift-csv", str(_abs(drift))])
        report = _canonical.write_canonical_report(phase, out_dir)
    elif phase == "phase0_drift_diagnostic":
        print("[summarize] phase0 drift report is produced by the runner itself; nothing to do.")
        return
    else:
        raise SystemExit(f"No summarizer for phase '{phase}'.")
    if report is not None:
        print(f"[summarize] canonical report -> {report}")
