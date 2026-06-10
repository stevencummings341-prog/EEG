#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregate within/cross-session baseline runs into the full 5-seed evaluation report.

Reads the per-run CSVs written by scripts/train_session_models.py
(``<results_dir>/runs/{protocol}__{model}__seed{seed}.csv``; also accepts the legacy
``{protocol}__{model}.csv`` from job 21161 — the seed is read from the row column, so
filename style does not matter) and produces, under ``<results_dir>/summaries/``:

  results_within_session.csv        all raw within rows (model,subject,session,seed,fold,...)
  results_cross_session.csv         all raw cross rows (model,subject,train->test,seed,...)
  within_by_seed.csv                per (model,seed) within accuracy etc. (folds+sessions collapsed)
  cross_by_seed.csv                 per (model,seed) cross accuracy etc.
  within_session_wise.csv           per (model,session) mean+/-std across seeds (ses-01/02/03)
  cross_by_direction.csv            per (model,train->test) mean+/-std across seeds
  summary_by_model_protocol.csv     per (model,protocol) mean+/-std/median/min/max across seeds
  within_session_accuracy_boxplot.png
  cross_session_accuracy_matrix_by_model.png
  protocol_comparison.png
  model_ranking.md
  SESSION_MODEL_COMPARE_REPORT.md   the full report (incl. paper comparison + reliability)

Aggregation: metrics are first averaged within an evaluation unit (within: collapse the
10 folds per session; cross: per directed pair), then averaged across units per seed to
get one score per (model, seed), then mean+/-std/median/min/max ACROSS seeds — i.e. the
reported "mean +/- std across N seeds". CPU-only (pandas/matplotlib).

NaN-safe and partial-data-safe: if some seeds/protocols are missing it still runs and the
report flags what is incomplete (the dependent job marks the report INCOMPLETE accordingly).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRIC_COLS = ["accuracy", "balanced_accuracy", "macro_f1", "auc", "nll", "brier", "ece"]
WITHIN_UNIT = ["model", "subject", "session"]      # collapse folds within (unit, seed)
CROSS_UNIT = ["model", "subject", "train_session", "test_session"]
MODEL_PREF = ["eegnet", "deepconvnet", "fbcnet", "cap_eegnet"]

# WBCIC-SHU paper EEGNet/DeepConvNet/FBCNet within-session 10-fold CV baselines (%),
# plus the EEGNet per-session learning-effect trend. Reference only.
PAPER_WITHIN_ACC = {"eegnet": 85.32, "deepconvnet": 84.47, "fbcnet": 78.40}
PAPER_SESSION_TREND = {"ses-01": 81.77, "ses-02": 86.63, "ses-03": 88.90}

N_OK_SESSIONS = 148
N_WITHIN_FOLDS = 10
N_CROSS_PAIRS = 288   # directed, all subjects with >=2 ok sessions (47x6 + 3x2)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _load_runs(runs_dir: Path) -> pd.DataFrame:
    files = sorted(runs_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No run CSVs in {runs_dir}. Run scripts/train_session_models.py first.")
    frames = []
    for fp in files:
        try:
            d = pd.read_csv(fp)
            if len(d):
                frames.append(d)
        except pd.errors.EmptyDataError:
            continue
    if not frames:
        raise ValueError(f"All run CSVs in {runs_dir} are empty.")
    df = pd.concat(frames, ignore_index=True)
    # de-duplicate in case a legacy + seed-suffixed file overlap on identical rows
    key = [c for c in ["model", "protocol", "subject", "session", "train_session",
                       "test_session", "seed", "fold"] if c in df.columns]
    df = df.drop_duplicates(subset=key)
    return df


def _model_order(models) -> List[str]:
    models = list(models)
    return [m for m in MODEL_PREF if m in models] + [m for m in models if m not in MODEL_PREF]


def _ses_int(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"(\d+)").iloc[:, 0].astype(float)


# --------------------------------------------------------------------------- #
# Per-seed aggregation
# --------------------------------------------------------------------------- #
def _unit_means(df: pd.DataFrame, unit_keys: List[str]) -> pd.DataFrame:
    """Average metrics within (unit_keys + seed) -> one row per evaluation unit per seed."""
    present = [c for c in METRIC_COLS if c in df.columns]
    keys = unit_keys + ["seed"]
    return df.groupby(keys, as_index=False)[present].mean()


def _per_model_seed(unit_df: pd.DataFrame) -> pd.DataFrame:
    """Average evaluation units within (model, seed) -> one score per (model, seed)."""
    present = [c for c in METRIC_COLS if c in unit_df.columns]
    g = unit_df.groupby(["model", "seed"], as_index=False)[present].mean()
    return g


def _across_seeds(per_seed: pd.DataFrame, protocol: str) -> pd.DataFrame:
    """mean/std/median/min/max ACROSS seeds, per model."""
    present = [c for c in METRIC_COLS if c in per_seed.columns]
    rows = []
    for model, g in per_seed.groupby("model"):
        row = {"model": model, "protocol": protocol,
               "n_seeds": int(g["seed"].nunique())}
        for c in present:
            v = g[c].dropna().values
            row[f"{c}_mean"] = float(np.mean(v)) if len(v) else np.nan
            row[f"{c}_std"] = float(np.std(v, ddof=0)) if len(v) else np.nan
            row[f"{c}_median"] = float(np.median(v)) if len(v) else np.nan
            row[f"{c}_min"] = float(np.min(v)) if len(v) else np.nan
            row[f"{c}_max"] = float(np.max(v)) if len(v) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _within_session_wise(within_df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, session): mean+/-std of accuracy across seeds (subjects+folds collapsed)."""
    present = [c for c in METRIC_COLS if c in within_df.columns]
    # collapse folds -> (model,subject,session,seed); then over subjects -> (model,session,seed)
    u = within_df.groupby(["model", "subject", "session", "seed"], as_index=False)[present].mean()
    ms = u.groupby(["model", "session", "seed"], as_index=False)[present].mean()
    rows = []
    for (model, session), g in ms.groupby(["model", "session"]):
        row = {"model": model, "session": session, "n_seeds": int(g["seed"].nunique())}
        for c in present:
            v = g[c].dropna().values
            row[f"{c}_mean"] = float(np.mean(v)) if len(v) else np.nan
            row[f"{c}_std"] = float(np.std(v, ddof=0)) if len(v) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _cross_by_direction(cross_df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, train_session, test_session): mean+/-std accuracy across seeds."""
    present = [c for c in METRIC_COLS if c in cross_df.columns]
    u = cross_df.groupby(["model", "train_session", "test_session", "seed"],
                         as_index=False)[present].mean()
    rows = []
    for (model, tr, te), g in u.groupby(["model", "train_session", "test_session"]):
        row = {"model": model, "train_session": tr, "test_session": te,
               "n_seeds": int(g["seed"].nunique())}
        for c in present:
            v = g[c].dropna().values
            row[f"{c}_mean"] = float(np.mean(v)) if len(v) else np.nan
            row[f"{c}_std"] = float(np.std(v, ddof=0)) if len(v) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_within_boxplot(within_unit: pd.DataFrame, out: Path) -> None:
    models = _model_order(within_unit["model"].unique())
    data = [within_unit.loc[within_unit["model"] == m, "accuracy"].dropna().values for m in models]
    fig, ax = plt.subplots(figsize=(1.6 * len(models) + 3, 5))
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(models) + 1)); ax.set_xticklabels(models)
    ax.axhline(0.5, color="gray", linestyle=":", label="chance")
    ax.set_ylabel("within-session accuracy (per subject-session, seeds pooled)")
    ax.set_title("Within-session 10-fold CV accuracy by model")
    ax.legend(); fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_cross_matrix(cross_df: pd.DataFrame, out: Path) -> None:
    models = _model_order(cross_df["model"].unique())
    c = cross_df.copy()
    c["tr"] = _ses_int(c["train_session"]); c["te"] = _ses_int(c["test_session"])
    sess = sorted(set(c["tr"].dropna()).union(set(c["te"].dropna())))
    n = len(sess); idx = {s: i for i, s in enumerate(sess)}
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models) + 1, 4.2), squeeze=False)
    for ax, model in zip(axes[0], models):
        mat = np.full((n, n), np.nan)
        piv = c[c["model"] == model].groupby(["tr", "te"])["accuracy"].mean()
        for (tr, te), v in piv.items():
            mat[idx[tr], idx[te]] = v
        im = ax.imshow(mat, vmin=0.5, vmax=1.0, cmap="YlOrRd")
        ax.set_xticks(range(n)); ax.set_xticklabels([f"S{int(s)}" for s in sess])
        ax.set_yticks(range(n)); ax.set_yticklabels([f"S{int(s)}" for s in sess])
        ax.set_xlabel("test session"); ax.set_ylabel("train session"); ax.set_title(model)
        for i in range(n):
            for j in range(n):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Cross-session accuracy (train -> test, mean over subjects + seeds)")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_protocol_comparison(within_across: pd.DataFrame, cross_across: pd.DataFrame, out: Path) -> None:
    models = _model_order(set(within_across["model"]).union(cross_across["model"]))
    wi = within_across.set_index("model") if len(within_across) else pd.DataFrame()
    cr = cross_across.set_index("model") if len(cross_across) else pd.DataFrame()
    x = np.arange(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(1.7 * len(models) + 3, 5))
    if len(wi):
        ax.bar(x - w / 2, [wi.loc[m, "accuracy_mean"] if m in wi.index else np.nan for m in models],
               w, yerr=[wi.loc[m, "accuracy_std"] if m in wi.index else 0 for m in models],
               capsize=4, label="within-session")
    if len(cr):
        ax.bar(x + w / 2, [cr.loc[m, "accuracy_mean"] if m in cr.index else np.nan for m in models],
               w, yerr=[cr.loc[m, "accuracy_std"] if m in cr.index else 0 for m in models],
               capsize=4, label="cross-session")
    ax.axhline(0.5, color="gray", linestyle=":", label="chance")
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("accuracy (mean +/- std across seeds)")
    ax.set_title("Within vs Cross-session accuracy by model")
    ax.legend(); fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


# --------------------------------------------------------------------------- #
# Reliability checks
# --------------------------------------------------------------------------- #
def reliability(within_df: pd.DataFrame, cross_df: pd.DataFrame) -> Dict[str, object]:
    rep: Dict[str, object] = {}
    # completed (model, protocol, seed) cells
    cells = []
    for proto, d in (("within", within_df), ("cross", cross_df)):
        if len(d):
            for (m, s), _ in d.groupby(["model", "seed"]):
                cells.append({"protocol": proto, "model": m, "seed": int(s)})
    rep["completed_cells"] = cells
    rep["n_completed_cells"] = len(cells)

    # within: expected rows per (model,seed) = 148*10; check sessions used
    rep["within"] = {}
    if len(within_df):
        wr = []
        for (m, s), g in within_df.groupby(["model", "seed"]):
            n_sessions = g[["subject", "session"]].drop_duplicates().shape[0]
            wr.append({"model": m, "seed": int(s), "rows": int(len(g)),
                       "expected_rows": N_OK_SESSIONS * N_WITHIN_FOLDS,
                       "n_sessions": int(n_sessions),
                       "complete": bool(len(g) == N_OK_SESSIONS * N_WITHIN_FOLDS)})
        rep["within"]["per_cell"] = wr
        rep["within"]["distinct_sessions_overall"] = int(
            within_df[["subject", "session"]].drop_duplicates().shape[0])
        rep["within"]["all_148_sessions_used"] = bool(
            rep["within"]["distinct_sessions_overall"] == N_OK_SESSIONS)
    # cross: expected pairs per (model,seed) = 288; check directed validity
    rep["cross"] = {}
    if len(cross_df):
        cr = []
        for (m, s), g in cross_df.groupby(["model", "seed"]):
            cr.append({"model": m, "seed": int(s), "rows": int(len(g)),
                       "expected_rows": N_CROSS_PAIRS,
                       "complete": bool(len(g) == N_CROSS_PAIRS)})
        rep["cross"]["per_cell"] = cr
        bad = cross_df[cross_df["train_session"] == cross_df["test_session"]]
        rep["cross"]["invalid_same_session_pairs"] = int(len(bad))
        rep["cross"]["distinct_directed_pairs_overall"] = int(
            cross_df[["subject", "train_session", "test_session"]].drop_duplicates().shape[0])
    # NaN metrics + collapse (near-chance) flags
    nan_counts = {}
    for proto, d in (("within", within_df), ("cross", cross_df)):
        for c in METRIC_COLS:
            if len(d) and c in d.columns:
                nan_counts[f"{proto}.{c}"] = int(d[c].isna().sum())
    rep["nan_counts"] = nan_counts
    return rep


# --------------------------------------------------------------------------- #
# Report writers
# --------------------------------------------------------------------------- #
def _fmt_ms(row, metric):
    mean = row.get(f"{metric}_mean", np.nan)
    std = row.get(f"{metric}_std", np.nan)
    if pd.isna(mean):
        return "—"
    return f"{mean:.3f}±{std:.3f}" if not pd.isna(std) else f"{mean:.3f}"


def write_ranking(within_across, cross_across, out: Path) -> pd.DataFrame:
    models = _model_order(set(within_across["model"]).union(cross_across["model"]))
    wi = within_across.set_index("model") if len(within_across) else pd.DataFrame()
    cr = cross_across.set_index("model") if len(cross_across) else pd.DataFrame()
    rows = []
    for m in models:
        wa = float(wi.loc[m, "accuracy_mean"]) if (len(wi) and m in wi.index) else np.nan
        ca = float(cr.loc[m, "accuracy_mean"]) if (len(cr) and m in cr.index) else np.nan
        drop = wa - ca if (not np.isnan(wa) and not np.isnan(ca)) else np.nan
        rel = (1 - ca / wa) if (not np.isnan(wa) and wa > 0 and not np.isnan(ca)) else np.nan
        rows.append({"model": m, "within_acc": wa, "cross_acc": ca,
                     "cross_session_drop": drop, "relative_drop": rel})
    drop_df = pd.DataFrame(rows)

    lines = ["# Model ranking — within vs cross-session (baselines, 5-seed)\n"]
    if len(cr):
        rk = drop_df.dropna(subset=["cross_acc"]).sort_values("cross_acc", ascending=False)
        lines.append("**Ranked by cross-session accuracy:**\n")
        for i, r in enumerate(rk.itertuples(), 1):
            lines.append(f"{i}. `{r.model}` — cross {r.cross_acc:.4f}, within {r.within_acc:.4f}, "
                         f"drop {r.cross_session_drop:.4f} ({r.relative_drop*100:.1f}%)")
        lines.append("")
    if len(wi):
        rk = drop_df.dropna(subset=["within_acc"]).sort_values("within_acc", ascending=False)
        lines.append("**Ranked by within-session accuracy:**\n")
        for i, r in enumerate(rk.itertuples(), 1):
            lines.append(f"{i}. `{r.model}` — within {r.within_acc:.4f}")
        lines.append("")
    lines.append("## Cross-session drop table\n")
    lines.append("| model | within Acc | cross Acc | drop (abs) | relative drop |")
    lines.append("|---|---|---|---|---|")
    for r in drop_df.itertuples():
        def f(v): return "—" if (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"
        rel = "—" if np.isnan(r.relative_drop) else f"{r.relative_drop*100:.1f}%"
        lines.append(f"| `{r.model}` | {f(r.within_acc)} | {f(r.cross_acc)} | "
                     f"{f(r.cross_session_drop)} | {rel} |")
    lines.append("\n> drop = within mean Acc − cross mean Acc; relative drop = 1 − Acc_cross/Acc_within.\n")
    out.write_text("\n".join(lines), encoding="utf-8")
    return drop_df


def write_report(within_across, cross_across, within_sw, cross_dir, drop_df, rel,
                 figures, out: Path, run_id: str, incomplete: bool) -> None:
    L: List[str] = []
    L.append(f"# Session Model Comparison Report — {run_id}\n")
    if incomplete:
        L.append("> ⚠️ **INCOMPLETE**: not all expected runs are present (see Reliability). "
                 "Numbers below are computed on the available data only.\n")
    L.append("Baselines: **EEGNet / DeepConvNet / FBCNet** under one protocol, data filter "
             "(status=ok, 148 sessions), and metric set. Seeds aggregated as mean±std.\n")

    # 1. within mean±std across seeds
    L.append("## 1. Within-session 10-fold CV (mean ± std across seeds)\n")
    L.append("| model | n_seeds | Acc | BalAcc | MacroF1 | AUC | NLL | Brier | ECE | median Acc | min | max |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in within_across.sort_values("model").itertuples():
        d = r._asdict()
        L.append(f"| `{d['model']}` | {d.get('n_seeds','—')} | {_fmt_ms(d,'accuracy')} | "
                 f"{_fmt_ms(d,'balanced_accuracy')} | {_fmt_ms(d,'macro_f1')} | {_fmt_ms(d,'auc')} | "
                 f"{_fmt_ms(d,'nll')} | {_fmt_ms(d,'brier')} | {_fmt_ms(d,'ece')} | "
                 f"{d.get('accuracy_median',float('nan')):.3f} | {d.get('accuracy_min',float('nan')):.3f} | "
                 f"{d.get('accuracy_max',float('nan')):.3f} |")
    # session-wise
    if len(within_sw):
        L.append("\n### Within-session by session (Acc mean ± std across seeds)\n")
        L.append("| model | ses-01 | ses-02 | ses-03 |")
        L.append("|---|---|---|---|")
        for model, g in within_sw.groupby("model"):
            gi = g.set_index("session")
            def cell(ses):
                if ses in gi.index:
                    return f"{gi.loc[ses,'accuracy_mean']:.3f}±{gi.loc[ses,'accuracy_std']:.3f}"
                return "—"
            L.append(f"| `{model}` | {cell('ses-01')} | {cell('ses-02')} | {cell('ses-03')} |")

    # 2. cross mean±std
    L.append("\n## 2. Cross-session (mean ± std across seeds)\n")
    L.append("| model | n_seeds | Acc | BalAcc | MacroF1 | AUC | NLL | Brier | ECE | median Acc | min | max |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in cross_across.sort_values("model").itertuples():
        d = r._asdict()
        L.append(f"| `{d['model']}` | {d.get('n_seeds','—')} | {_fmt_ms(d,'accuracy')} | "
                 f"{_fmt_ms(d,'balanced_accuracy')} | {_fmt_ms(d,'macro_f1')} | {_fmt_ms(d,'auc')} | "
                 f"{_fmt_ms(d,'nll')} | {_fmt_ms(d,'brier')} | {_fmt_ms(d,'ece')} | "
                 f"{d.get('accuracy_median',float('nan')):.3f} | {d.get('accuracy_min',float('nan')):.3f} | "
                 f"{d.get('accuracy_max',float('nan')):.3f} |")
    if len(cross_dir):
        L.append("\n### Cross-session by direction (Acc mean ± std across seeds)\n")
        L.append("| model | train → test | Acc |")
        L.append("|---|---|---|")
        for r in cross_dir.sort_values(["model", "train_session", "test_session"]).itertuples():
            L.append(f"| `{r.model}` | {r.train_session} → {r.test_session} | "
                     f"{r.accuracy_mean:.3f}±{r.accuracy_std:.3f} |")

    # 3. within vs cross
    L.append("\n## 3. Within vs Cross-session\n")
    L.append("| model | within Acc | cross Acc | drop | relative drop |")
    L.append("|---|---|---|---|---|")
    for r in drop_df.itertuples():
        def f(v): return "—" if (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"
        reld = "—" if np.isnan(r.relative_drop) else f"{r.relative_drop*100:.1f}%"
        L.append(f"| `{r.model}` | {f(r.within_acc)} | {f(r.cross_acc)} | "
                 f"{f(r.cross_session_drop)} | {reld} |")

    # 4. paper baseline comparison
    L.append("\n## 4. Paper baseline comparison (within-session 10-fold CV)\n")
    L.append("WBCIC-SHU paper reports (within-session 10-fold CV accuracy %):"
             " EEGNet 85.32, DeepConvNet 84.47, FBCNet 78.40.\n")
    L.append("| model | ours within Acc (%) | paper (%) | Δ (ours − paper, pp) |")
    L.append("|---|---|---|---|")
    wi = within_across.set_index("model")
    for m in _model_order(within_across["model"].unique()):
        ours = float(wi.loc[m, "accuracy_mean"]) * 100 if m in wi.index else np.nan
        paper = PAPER_WITHIN_ACC.get(m)
        if paper is None:
            continue
        delta = ours - paper if not np.isnan(ours) else np.nan
        oc = "—" if np.isnan(ours) else f"{ours:.2f}"
        dc = "—" if np.isnan(delta) else f"{delta:+.2f}"
        L.append(f"| `{m}` | {oc} | {paper:.2f} | {dc} |")
    if len(within_sw):
        L.append("\n**Session trend (EEGNet, ours vs paper, Acc %):** "
                 "paper S1 81.77 / S2 86.63 / S3 88.90.\n")
        eg = within_sw[within_sw["model"] == "eegnet"].set_index("session")
        L.append("| session | ours (%) | paper (%) |")
        L.append("|---|---|---|")
        for ses in ["ses-01", "ses-02", "ses-03"]:
            ours = f"{eg.loc[ses,'accuracy_mean']*100:.2f}" if ses in eg.index else "—"
            L.append(f"| {ses} | {ours} | {PAPER_SESSION_TREND[ses]:.2f} |")

    # 5. reliability
    L.append("\n## 5. Reliability checks\n")
    L.append(f"- Completed (model, protocol, seed) cells: **{rel['n_completed_cells']}** "
             f"(target 30 = 3 models × 2 protocols × 5 seeds).")
    if rel.get("within"):
        w = rel["within"]
        L.append(f"- Within: distinct (subject,session) used = "
                 f"**{w.get('distinct_sessions_overall','—')}** "
                 f"(all 148 used: {w.get('all_148_sessions_used','—')}); "
                 f"expected rows per (model,seed) = {N_OK_SESSIONS*N_WITHIN_FOLDS}.")
        incs = [c for c in w.get("per_cell", []) if not c["complete"]]
        if incs:
            L.append(f"  - ⚠️ incomplete within cells: "
                     + ", ".join(f"{c['model']}/seed{c['seed']}({c['rows']}/{c['expected_rows']})" for c in incs))
    if rel.get("cross"):
        c = rel["cross"]
        L.append(f"- Cross: distinct directed pairs = "
                 f"**{c.get('distinct_directed_pairs_overall','—')}** "
                 f"(expected {N_CROSS_PAIRS}); invalid same-session pairs = "
                 f"{c.get('invalid_same_session_pairs',0)} (must be 0 — no leakage).")
        incs = [x for x in c.get("per_cell", []) if not x["complete"]]
        if incs:
            L.append(f"  - ⚠️ incomplete cross cells: "
                     + ", ".join(f"{x['model']}/seed{x['seed']}({x['rows']}/{x['expected_rows']})" for x in incs))
    nz = {k: v for k, v in rel.get("nan_counts", {}).items() if v}
    L.append(f"- NaN metric cells: {nz if nz else 'none'}.")
    L.append("- Leakage: by construction within=disjoint folds (val carved from train only), "
             "cross=different sessions; no test trials in train/val.")

    L.append("\n## 6. Figures\n")
    for f in figures:
        L.append(f"- `{f}`")
    L.append("\n## Notes\n")
    L.append("- Mainline = the THREE baseline architectures. CAP-EEGNet (v1/v2) and all "
             "agent/toolkit/prototype/confidence/online/fine-tuning modules remain FUTURE work.")
    L.append("- LOSO and 41/10 are not run.")
    L.append("")
    out.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize within/cross-session baseline results.")
    ap.add_argument("--results-dir", default="outputs/experiments/session_model_compare_v1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    results_dir = results_dir if results_dir.is_absolute() else (PROJECT_ROOT / results_dir)
    runs_dir = results_dir / "runs"
    out_dir = Path(args.out) if args.out else (results_dir / "summaries")
    out_dir = out_dir if out_dir.is_absolute() else (PROJECT_ROOT / out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or results_dir.name

    df = _load_runs(runs_dir)
    within = df[df["protocol"] == "within_session"].copy()
    cross = df[df["protocol"] == "cross_session"].copy()
    print(f"[summarize] {len(df)} rows | within={len(within)} cross={len(cross)} "
          f"| models={sorted(df['model'].unique())} | seeds={sorted(df['seed'].unique())}")

    if len(within):
        within.to_csv(out_dir / "results_within_session.csv", index=False, float_format="%.6f")
    if len(cross):
        cross.to_csv(out_dir / "results_cross_session.csv", index=False, float_format="%.6f")

    # per-seed + across-seed aggregates
    within_unit = _unit_means(within, WITHIN_UNIT) if len(within) else pd.DataFrame()
    cross_unit = _unit_means(cross, CROSS_UNIT) if len(cross) else pd.DataFrame()
    within_seed = _per_model_seed(within_unit) if len(within_unit) else pd.DataFrame()
    cross_seed = _per_model_seed(cross_unit) if len(cross_unit) else pd.DataFrame()
    within_across = _across_seeds(within_seed, "within_session") if len(within_seed) else pd.DataFrame()
    cross_across = _across_seeds(cross_seed, "cross_session") if len(cross_seed) else pd.DataFrame()
    within_sw = _within_session_wise(within) if len(within) else pd.DataFrame()
    cross_dir = _cross_by_direction(cross) if len(cross) else pd.DataFrame()

    if len(within_seed):
        within_seed.to_csv(out_dir / "within_by_seed.csv", index=False, float_format="%.6f")
    if len(cross_seed):
        cross_seed.to_csv(out_dir / "cross_by_seed.csv", index=False, float_format="%.6f")
    if len(within_sw):
        within_sw.to_csv(out_dir / "within_session_wise.csv", index=False, float_format="%.6f")
    if len(cross_dir):
        cross_dir.to_csv(out_dir / "cross_by_direction.csv", index=False, float_format="%.6f")
    summary = pd.concat([within_across, cross_across], ignore_index=True) if (
        len(within_across) or len(cross_across)) else pd.DataFrame()
    if len(summary):
        summary.to_csv(out_dir / "summary_by_model_protocol.csv", index=False, float_format="%.6f")

    figures: List[str] = []
    if len(within_unit):
        fig_within_boxplot(within_unit, out_dir / "within_session_accuracy_boxplot.png")
        figures.append("within_session_accuracy_boxplot.png")
    if len(cross):
        fig_cross_matrix(cross, out_dir / "cross_session_accuracy_matrix_by_model.png")
        figures.append("cross_session_accuracy_matrix_by_model.png")
    if len(within_across) or len(cross_across):
        fig_protocol_comparison(within_across, cross_across, out_dir / "protocol_comparison.png")
        figures.append("protocol_comparison.png")

    rel = reliability(within, cross)
    # incomplete if fewer than the full 3x2x5 grid OR any cell short of expected rows
    incomplete = rel["n_completed_cells"] < 30
    for sec in ("within", "cross"):
        for c in rel.get(sec, {}).get("per_cell", []):
            if not c["complete"]:
                incomplete = True

    drop_df = write_ranking(within_across, cross_across, out_dir / "model_ranking.md")
    write_report(within_across, cross_across, within_sw, cross_dir, drop_df, rel,
                 figures, out_dir / "SESSION_MODEL_COMPARE_REPORT.md", run_id, incomplete)
    print(f"[summarize] wrote tables + {len(figures)} figures + report to {out_dir} "
          f"| incomplete={incomplete}")


if __name__ == "__main__":
    main()
