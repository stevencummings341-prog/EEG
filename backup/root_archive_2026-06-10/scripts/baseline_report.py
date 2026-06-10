#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dependent overnight orchestrator: check training jobs, summarize, update docs.

Designed to run as a Slurm dependency (afterany) AFTER all baseline training jobs,
so it executes even if some training jobs failed and can report incomplete/failed runs.

Steps:
  1. Read the submitted job IDs (id protocol seed) from the job-ids file.
  2. Query sacct for each job's final state (COMPLETED/FAILED/CANCELLED/TIMEOUT/...).
  3. Run scripts/summarize_session_results.py if any result CSVs exist.
  4. Check whether expected result files exist; detect incomplete cells.
  5. Write summaries/RUN_STATUS.md (job table + file checks + INCOMPLETE flag).
  6. Append a dated status block to docs/EXPERIMENT_LOG.md and insert one into
     docs/PROGRESS.md (after the AUTORUN sentinel). Clearly marks the report INCOMPLETE
     if any training job did not COMPLETE or any expected result file is missing.

CPU-only; no torch. Never touches raw / workspace2.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SENTINEL = "<!-- AUTORUN_STATUS_BELOW: baseline_report.py inserts entries here -->"
GOOD_STATE = "COMPLETED"


def read_job_ids(path: Path) -> List[Dict[str, str]]:
    """Read 'jobid protocol seed' lines."""
    jobs = []
    if not path.exists():
        return jobs
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit():
            jobs.append({"job_id": parts[0], "protocol": parts[1], "seed": parts[2]})
    return jobs


def sacct_state(job_id: str) -> Tuple[str, str, str]:
    """Return (state, elapsed, exit_code) for the main job step via sacct."""
    try:
        out = subprocess.run(
            ["sacct", "-j", job_id, "--format=JobID,State,Elapsed,ExitCode",
             "--parsable2", "--noheader"],
            capture_output=True, text=True, timeout=60,
        ).stdout
    except Exception as e:  # noqa: BLE001
        return (f"SACCT_ERROR({e})", "", "")
    for line in out.splitlines():
        f = line.split("|")
        # main step has JobID == job_id (no .batch/.extern suffix)
        if f and f[0] == job_id:
            state = f[1].split()[0] if len(f) > 1 and f[1] else "UNKNOWN"
            elapsed = f[2] if len(f) > 2 else ""
            exitc = f[3] if len(f) > 3 else ""
            return (state, elapsed, exitc)
    return ("NOT_FOUND", "", "")


def run_summarizer(results_dir: Path, run_id: str) -> Tuple[bool, str]:
    """Run the summarizer; return (ok, combined_output)."""
    runs_dir = results_dir / "runs"
    csvs = list(runs_dir.glob("*.csv")) if runs_dir.exists() else []
    if not csvs:
        return (False, f"no result CSVs in {runs_dir}")
    try:
        r = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "summarize_session_results.py"),
             "--results-dir", str(results_dir), "--run-id", run_id],
            capture_output=True, text=True, timeout=1800,
        )
        return (r.returncode == 0, (r.stdout or "") + (r.stderr or ""))
    except Exception as e:  # noqa: BLE001
        return (False, f"summarizer error: {e}")


def expected_files(summaries: Path) -> List[Tuple[str, bool]]:
    names = [
        "results_within_session.csv", "results_cross_session.csv",
        "within_by_seed.csv", "cross_by_seed.csv",
        "within_session_wise.csv", "cross_by_direction.csv",
        "summary_by_model_protocol.csv", "model_ranking.md",
        "SESSION_MODEL_COMPARE_REPORT.md",
        "within_session_accuracy_boxplot.png",
        "cross_session_accuracy_matrix_by_model.png", "protocol_comparison.png",
    ]
    return [(n, (summaries / n).exists()) for n in names]


def write_run_status(path: Path, jobs: List[Dict], states: Dict[str, Tuple[str, str, str]],
                     files: List[Tuple[str, bool]], summarizer_ok: bool,
                     summarizer_log: str, incomplete: bool) -> None:
    L: List[str] = []
    L.append(f"# Overnight baseline run status — {datetime.now().isoformat(timespec='seconds')}\n")
    L.append(f"**Overall: {'INCOMPLETE / NEEDS ATTENTION' if incomplete else 'COMPLETE'}**\n")
    L.append("## Training jobs (sacct)\n")
    L.append("| job_id | protocol | seed | state | elapsed | exit |")
    L.append("|---|---|---|---|---|---|")
    for j in jobs:
        st, el, ec = states.get(j["job_id"], ("?", "", ""))
        L.append(f"| {j['job_id']} | {j['protocol']} | {j['seed']} | {st} | {el} | {ec} |")
    bad = [j["job_id"] for j in jobs if states.get(j["job_id"], ("",))[0] != GOOD_STATE]
    L.append(f"\nNon-COMPLETED jobs: {bad if bad else 'none'}.\n")
    L.append("## Summarizer\n")
    L.append(f"- ran OK: **{summarizer_ok}**")
    L.append("```")
    L.append(summarizer_log.strip()[-1500:])
    L.append("```")
    L.append("\n## Expected output files\n")
    L.append("| file | present |")
    L.append("|---|---|")
    for n, ok in files:
        L.append(f"| `{n}` | {'yes' if ok else 'MISSING'} |")
    L.append("\nSee `SESSION_MODEL_COMPARE_REPORT.md` for the full metrics report.\n")
    path.write_text("\n".join(L), encoding="utf-8")


def append_experiment_log(path: Path, jobs, states, incomplete: bool, results_dir: Path) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    bad = [f"{j['job_id']}({j['protocol']}/s{j['seed']}:{states.get(j['job_id'],('?',))[0]})"
           for j in jobs if states.get(j["job_id"], ("",))[0] != GOOD_STATE]
    block = [
        f"\n## {ts} — overnight 5-seed baseline run (auto)\n",
        f"- Status: **{'INCOMPLETE' if incomplete else 'COMPLETE'}**.",
        f"- Models: EEGNet, DeepConvNet, FBCNet. Protocols: within-session CV + cross-session. "
        f"Seeds: 0–4.",
        f"- Jobs: {', '.join(j['job_id'] for j in jobs)}.",
        f"- Non-COMPLETED: {bad if bad else 'none'}.",
        f"- Results: `{results_dir}/summaries/` "
        f"(SESSION_MODEL_COMPARE_REPORT.md, RUN_STATUS.md, results_*.csv).",
        "",
    ]
    header = "" if path.exists() else (
        "# Experiment Log (project)\n\nNewest entries on top. Auto + manual entries.\n")
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text(header + "\n".join(block) + "\n" + existing, encoding="utf-8")


def insert_progress(path: Path, jobs, states, incomplete: bool, results_dir: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if SENTINEL not in text:
        return  # sentinel must be pre-placed; skip rather than corrupt the curated file
    ts = datetime.now().strftime("%Y-%m-%d")
    bad = [f"{j['job_id']}({j['protocol']}/s{j['seed']})"
           for j in jobs if states.get(j["job_id"], ("",))[0] != GOOD_STATE]
    entry = [
        f"\n## {ts} — overnight 5-seed baseline run completed (auto)\n",
        f"**Status: {'INCOMPLETE — see RUN_STATUS.md' if incomplete else 'COMPLETE'}.** "
        "Models EEGNet/DeepConvNet/FBCNet; protocols within-session CV + cross-session; seeds 0–4.",
        f"Jobs {', '.join(j['job_id'] for j in jobs)}; non-COMPLETED: {bad if bad else 'none'}.",
        f"Full report + tables + figures: `{results_dir}/summaries/` "
        "(`SESSION_MODEL_COMPARE_REPORT.md`, `RUN_STATUS.md`).",
        "",
    ]
    text = text.replace(SENTINEL, SENTINEL + "\n" + "\n".join(entry), 1)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Overnight baseline report orchestrator.")
    ap.add_argument("--results-dir", default="outputs/experiments/session_model_compare_v1")
    ap.add_argument("--job-ids-file", default=None)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    results_dir = results_dir if results_dir.is_absolute() else (PROJECT_ROOT / results_dir)
    job_ids_file = Path(args.job_ids_file) if args.job_ids_file else (results_dir / "overnight_job_ids.txt")
    run_id = args.run_id or results_dir.name
    summaries = results_dir / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)

    jobs = read_job_ids(job_ids_file)
    states = {j["job_id"]: sacct_state(j["job_id"]) for j in jobs}
    print("[report] job states:", {j["job_id"]: states[j["job_id"]][0] for j in jobs})

    summarizer_ok, slog = run_summarizer(results_dir, run_id)
    print(f"[report] summarizer ok={summarizer_ok}")

    files = expected_files(summaries)
    any_missing = any(not ok for _, ok in files)
    any_job_bad = any(states.get(j["job_id"], ("",))[0] != GOOD_STATE for j in jobs)
    incomplete = (not summarizer_ok) or any_missing or any_job_bad

    write_run_status(summaries / "RUN_STATUS.md", jobs, states, files,
                     summarizer_ok, slog, incomplete)
    append_experiment_log(PROJECT_ROOT / "docs" / "EXPERIMENT_LOG.md", jobs, states,
                          incomplete, results_dir)
    insert_progress(PROJECT_ROOT / "docs" / "PROGRESS.md", jobs, states, incomplete, results_dir)
    print(f"[report] DONE | incomplete={incomplete} | RUN_STATUS.md written")


if __name__ == "__main__":
    main()
