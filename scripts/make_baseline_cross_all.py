#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the alignment-compatible no-alignment baseline table for Phase 2b.

用途：Phase 2b 的 `code/summaries/alignment.py` 需要一个 `none_reference`（无对齐）
基线表，其 schema 为 `acc/bacc/f1 + train_sessions + training_scope`。但 Phase 1 的
`code/summaries/session.py` 产出的 cross 结果用的是 `accuracy/balanced_accuracy/
macro_f1 + train_session`（无 training_scope）。本脚本把后者转换成前者，写到
config 里 `output.baseline_cross_all` 指向的路径，供 Phase 2b summarize 做
alignment_vs_baseline（数据集无关，SHU/WBCIC 通用）。

输入：Phase 1 summarize 产出的 `summaries/results_cross_session.csv`
      （或用 --results-dir 指向 phase1 output_dir，自动回退到 runs/cross__*.csv）。
输出：`<...>/cross_session/tables/results_cross_session_all.csv`
      （单源有向对，training_scope=single_source）。

依赖: pandas >= 2.0
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 目标 schema（alignment.load_none_reference 直接引用的列）。
RENAME = {"accuracy": "acc", "balanced_accuracy": "bacc", "macro_f1": "f1",
          "train_session": "train_sessions"}
KEEP = ["training_scope", "method", "model", "seed", "subject", "train_sessions",
        "test_session", "acc", "bacc", "f1", "auc", "nll", "brier", "ece",
        "n_train", "n_val", "n_test", "checkpoint_path"]


def _abs(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _load_cross(cross_csv: Path | None, results_dir: Path | None) -> pd.DataFrame:
    """Load phase1 cross rows from the summaries CSV, else concat runs/cross__*.csv."""
    if cross_csv and cross_csv.exists():
        return pd.read_csv(cross_csv)
    if results_dir:
        summ = results_dir / "summaries" / "results_cross_session.csv"
        if summ.exists():
            return pd.read_csv(summ)
        runs = sorted(glob.glob(str(results_dir / "runs" / "cross__*.csv")))
        frames = [pd.read_csv(f) for f in runs if Path(f).stat().st_size > 0]
        if frames:
            df = pd.concat(frames, ignore_index=True)
            return df[df.get("protocol", "cross_session") == "cross_session"].copy()
    raise FileNotFoundError(
        "No cross-session source found. Pass --cross-csv or --results-dir with "
        "phase1 summaries/results_cross_session.csv (run phase1 --summarize first).")


def build(cross_csv: Path | None, results_dir: Path | None, out: Path) -> pd.DataFrame:
    df = _load_cross(cross_csv, results_dir)
    df = df.rename(columns=RENAME).copy()
    df["training_scope"] = "single_source"
    df["method"] = "none_reference"
    if "checkpoint_path" not in df.columns:
        df["checkpoint_path"] = ""
    for col in ["acc", "bacc", "f1", "auc", "nll", "brier", "ece",
                "n_train", "n_val", "n_test"]:
        if col not in df.columns:
            df[col] = pd.NA
    out.parent.mkdir(parents=True, exist_ok=True)
    keep = [c for c in KEEP if c in df.columns]
    df[keep].to_csv(out, index=False, float_format="%.6f")
    return df[keep]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build alignment-compatible baseline_cross_all table.")
    ap.add_argument("--cross-csv", default=None,
                    help="phase1 summaries/results_cross_session.csv (preferred)")
    ap.add_argument("--results-dir", default=None,
                    help="phase1 output_dir (fallback: summaries/ then runs/cross__*.csv)")
    ap.add_argument("--out", required=True,
                    help="target results_cross_session_all.csv path")
    args = ap.parse_args()
    cross_csv = _abs(args.cross_csv) if args.cross_csv else None
    results_dir = _abs(args.results_dir) if args.results_dir else None
    out = _abs(args.out)
    df = build(cross_csv, results_dir, out)
    print(f"[make_baseline_cross_all] wrote {len(df)} rows -> {out}")
    print(f"[make_baseline_cross_all] models={sorted(df['model'].unique())} "
          f"seeds={sorted(df['seed'].unique())} scopes={sorted(df['training_scope'].unique())}")


if __name__ == "__main__":
    main()
