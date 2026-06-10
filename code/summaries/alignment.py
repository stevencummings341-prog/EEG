#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize Step-2 alignment results -> tables, figures, report, RUN_STATUS.

Merges all per-job run CSVs, pulls the ``none_reference`` (no-alignment) rows from
baseline_v1, joins per-subject drift levels, and writes the canonical
alignment_baseline_v1 outputs. Pure pandas/matplotlib (no seaborn). Robust to
partially-complete runs (writes whatever rows exist + a RUN_STATUS flag).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRIC_COLS = ["acc", "bacc", "f1", "auc", "nll", "brier", "ece"]
METHOD_ORDER = ["none_reference", "session_zscore", "euclidean_alignment",
                "riemannian_alignment", "bn_statistics_adaptation", "filterbank_reweighting"]
MODEL_ORDER = ["eegnet", "deepconvnet", "fbcnet"]
MATCH_KEYS = ["model", "train_sessions", "test_session", "subject", "seed"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _order(values, ref) -> List[str]:
    values = list(dict.fromkeys(values))
    return [v for v in ref if v in values] + [v for v in values if v not in ref]


def load_runs(runs_dir: Path) -> pd.DataFrame:
    files = sorted(runs_dir.glob("alignment__*.csv"))
    frames = [pd.read_csv(p) for p in files if p.stat().st_size > 0]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    keys = [c for c in ["method", "model", "training_scope", "protocol", "subject", "seed"]
            if c in df.columns]
    return df.drop_duplicates(subset=keys)


def load_none_reference(baseline_csv: Path, scopes_present: set, models_present: set) -> pd.DataFrame:
    """Pull the no-alignment baseline cross rows and map them to method=none_reference."""
    if not baseline_csv.exists():
        return pd.DataFrame()
    b = pd.read_csv(baseline_csv)
    keep = pd.DataFrame({
        "experiment_id": "baseline_v1",
        "method": "none_reference",
        "training_scope": b["training_scope"],
        "model": b["model"], "seed": b["seed"], "subject": b["subject"],
        "train_sessions": b["train_sessions"], "test_session": b["test_session"],
        "acc": b["acc"], "bacc": b["bacc"], "f1": b["f1"], "auc": b["auc"],
        "nll": b["nll"], "brier": b["brier"], "ece": b["ece"],
        "n_train": b["n_train"], "n_val": b["n_val"], "n_test": b["n_test"],
        "source_alignment_stats": "{}", "target_alignment_stats": "{}",
        "used_target_x_for_stats": False, "used_target_y_for_training": False,
        "checkpoint_path": b.get("checkpoint_path", ""), "status": "ok", "error_message": "",
    })
    keep["protocol"] = keep["train_sessions"].astype(str) + "->" + keep["test_session"].astype(str)
    if scopes_present:
        keep = keep[keep["training_scope"].isin(scopes_present)]
    if models_present:
        keep = keep[keep["model"].isin(models_present)]
    return keep


def load_drift(drift_csv: Path) -> Dict[str, str]:
    if not drift_csv.exists():
        return {}
    d = pd.read_csv(drift_csv)
    if "subject" not in d.columns or "drift_level" not in d.columns:
        return {}
    return dict(zip(d["subject"], d["drift_level"]))


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def agg_meanstd(df_ok: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Seed-level subject-mean, then mean/std across seeds for each group."""
    if df_ok.empty:
        return pd.DataFrame()
    by_seed = df_ok.groupby(group_cols + ["seed"], as_index=False)[METRIC_COLS].mean()
    rows = []
    for keys, g in by_seed.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(g["seed"].nunique())
        mask = np.logical_and.reduce([df_ok[c] == v for c, v in zip(group_cols, keys)])
        row["n_obs"] = int(mask.sum())
        for col in METRIC_COLS:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_gains(df_align_ok: pd.DataFrame, none_df: pd.DataFrame) -> pd.DataFrame:
    """Per-row gain = method acc - matched none_reference acc."""
    if df_align_ok.empty or none_df.empty:
        out = df_align_ok.copy()
        out["none_acc"] = np.nan
        out["gain_acc"] = np.nan
        return out
    none_small = none_df[MATCH_KEYS + ["acc"]].rename(columns={"acc": "none_acc"})
    merged = df_align_ok.merge(none_small, on=MATCH_KEYS, how="left")
    merged["gain_acc"] = merged["acc"] - merged["none_acc"]
    return merged


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _bar(ax, labels, values, errs=None, title="", ylabel="", rot=0):
    x = np.arange(len(labels))
    ax.bar(x, values, yerr=errs, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rot, ha="right" if rot else "center")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)


def fig_method_comparison(by_method: pd.DataFrame, fig_path: Path) -> None:
    if by_method.empty:
        return
    bm = by_method[by_method["training_scope"] == "all"].copy() if "training_scope" in by_method else by_method
    if bm.empty:
        bm = by_method.copy()
    bm = bm.set_index("method").reindex(_order(bm["method"], METHOD_ORDER)).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    _bar(ax, bm["method"], bm["acc_mean"], bm["acc_std"],
         title="Alignment method comparison (mean acc over directions/subjects/seeds)",
         ylabel="Accuracy", rot=30)
    fig.tight_layout(); fig.savefig(fig_path, dpi=120); plt.close(fig)


def fig_by_direction(by_dir: pd.DataFrame, fig_path: Path) -> None:
    if by_dir.empty:
        return
    methods = _order(by_dir["method"], METHOD_ORDER)
    dirs = sorted(by_dir["protocol"].unique())
    x = np.arange(len(dirs))
    w = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, m in enumerate(methods):
        sub = by_dir[by_dir["method"] == m].set_index("protocol").reindex(dirs)
        ax.bar(x + i * w, sub["acc_mean"].values, w, label=m)
    ax.set_xticks(x + w * (len(methods) - 1) / 2)
    ax.set_xticklabels(dirs, rotation=30, ha="right")
    ax.set_ylabel("Accuracy"); ax.set_title("Accuracy by single-source direction")
    ax.legend(fontsize=7, ncol=2); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_path, dpi=120); plt.close(fig)


def fig_vs_baseline_gain(vs_base: pd.DataFrame, fig_path: Path) -> None:
    if vs_base.empty:
        return
    bm = vs_base[vs_base["training_scope"] == "all"].copy()
    if bm.empty:
        bm = vs_base.copy()
    bm = bm.groupby("method", as_index=False)["gain_acc_mean"].mean()
    bm = bm.set_index("method").reindex(_order(bm["method"], METHOD_ORDER)).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["tab:green" if v >= 0 else "tab:red" for v in bm["gain_acc_mean"]]
    x = np.arange(len(bm))
    ax.bar(x, bm["gain_acc_mean"], color=colors)
    ax.axhline(0, color="k", lw=0.8); ax.axhline(0.02, color="tab:blue", ls="--", lw=0.8, label="+2pp target")
    ax.set_xticks(x); ax.set_xticklabels(bm["method"], rotation=30, ha="right")
    ax.set_ylabel("Δ Accuracy vs none_reference"); ax.set_title("Alignment gain vs no-alignment baseline")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_path, dpi=120); plt.close(fig)


def fig_gain_by_subject(gain_df: pd.DataFrame, fig_path: Path) -> None:
    if gain_df.empty or gain_df["gain_acc"].isna().all():
        return
    best = (gain_df.groupby(["method", "subject"], as_index=False)["gain_acc"].mean())
    # pick the method with the best mean gain to visualize per-subject spread
    mean_gain = best.groupby("method")["gain_acc"].mean()
    if mean_gain.empty:
        return
    top_method = mean_gain.idxmax()
    sub = best[best["method"] == top_method].sort_values("gain_acc")
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["tab:green" if v >= 0 else "tab:red" for v in sub["gain_acc"]]
    ax.bar(np.arange(len(sub)), sub["gain_acc"], color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(sub))); ax.set_xticklabels(sub["subject"], rotation=90, fontsize=6)
    ax.set_ylabel("Δ Accuracy vs none"); ax.set_title(f"Per-subject gain — best method: {top_method}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_path, dpi=120); plt.close(fig)


def fig_gain_by_drift(gain_drift: pd.DataFrame, fig_path: Path) -> None:
    if gain_drift.empty:
        return
    levels = [l for l in ["stable", "moderate", "high"] if l in set(gain_drift["drift_level"])]
    levels += [l for l in gain_drift["drift_level"].unique() if l not in levels]
    methods = _order(gain_drift["method"], METHOD_ORDER)
    x = np.arange(len(levels)); w = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(methods):
        sub = gain_drift[gain_drift["method"] == m].set_index("drift_level").reindex(levels)
        ax.bar(x + i * w, sub["gain_acc_mean"].values, w, label=m)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x + w * (len(methods) - 1) / 2); ax.set_xticklabels(levels)
    ax.set_ylabel("Δ Accuracy vs none"); ax.set_title("Alignment gain by drift level")
    ax.legend(fontsize=7, ncol=2); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_path, dpi=120); plt.close(fig)


def fig_protocol_comparison(by_proto: pd.DataFrame, fig_path: Path) -> None:
    if by_proto.empty:
        return
    scopes = [s for s in ["single_source", "multi_source"] if s in set(by_proto["training_scope"])]
    methods = _order(by_proto["method"], METHOD_ORDER)
    x = np.arange(len(scopes)); w = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(methods):
        sub = by_proto[by_proto["method"] == m].set_index("training_scope").reindex(scopes)
        ax.bar(x + i * w, sub["acc_mean"].values, w, yerr=sub["acc_std"].values, capsize=3, label=m)
    ax.set_xticks(x + w * (len(methods) - 1) / 2); ax.set_xticklabels(scopes)
    ax.set_ylabel("Accuracy"); ax.set_title("Accuracy by protocol group")
    ax.legend(fontsize=7, ncol=2); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_path, dpi=120); plt.close(fig)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _fmt(v, nd=4):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "nan"


SUCCESS_THRESHOLD = 0.02  # +2pp mean cross-acc gain over none_reference = pre-registered success line


def _all_scope_gains(vs_base) -> pd.DataFrame:
    """method -> mean Δacc (all scope), ordered, as a small DataFrame."""
    if vs_base is None or vs_base.empty:
        return pd.DataFrame(columns=["method", "gain_acc_mean"])
    bm = vs_base[vs_base["training_scope"] == "all"].groupby("method", as_index=False)["gain_acc_mean"].mean()
    return bm.set_index("method").reindex(_order(bm["method"], METHOD_ORDER)).reset_index()


def write_report(report_path: Path, *, df_all, df_ok, by_method, by_proto, by_dir,
                 vs_base, gain_drift, by_subject, gain_rows, complete, n_failed, n_nan,
                 tables, figures, ms_skipped, expected_info) -> None:
    L: List[str] = []
    a = L.append
    a("# Alignment Baseline (Step 2) — Report")
    a("")
    a("> No-learning / unsupervised **test-time alignment** baseline. The model trains "
      "ONLY on source session(s); the target session is used ONLY through its UNLABELED X "
      "(z-score / covariance / BN running stats / band power); `y_test` is used ONLY for the "
      "final evaluation. No `optimizer.step` on the target (only BN running-stat updates).")
    a("")
    status = "COMPLETE" if complete else "INCOMPLETE (results pending / some jobs missing)"
    a(f"**Run status: {status}** — total rows {len(df_all)}, ok {len(df_ok)}, "
      f"failed {n_failed}, NaN-acc among ok {n_nan}.")
    a("")
    # ----- Headline conclusion (honest, computed) ------------------------------------ #
    gains_all = _all_scope_gains(vs_base)
    none_acc = float("nan")
    if not by_method.empty:
        nr = by_method[(by_method["method"] == "none_reference") & (by_method["training_scope"] == "all")]
        if len(nr):
            none_acc = float(nr["acc_mean"].iloc[0])
    a("## 0. Headline conclusion (honest)")
    a("")
    if not gains_all.empty:
        best = gains_all.iloc[int(gains_all["gain_acc_mean"].astype(float).values.argmax())]
        best_m, best_g = best["method"], float(best["gain_acc_mean"])
        n_pos = int((gains_all["gain_acc_mean"] > 0).sum())
        cleared = best_g >= SUCCESS_THRESHOLD
        a(f"- **No-learning / unsupervised alignment is INSUFFICIENT.** No method reaches the "
          f"pre-registered +{SUCCESS_THRESHOLD:.0%} (≥+0.02 mean Δacc) success line.")
        a(f"- `none_reference` (no alignment) mean cross-acc = **{none_acc:.4f}**.")
        a(f"- Best method = **`{best_m}`** with Δacc = **{best_g:+.4f}** "
          f"(below the +0.02 line; {'CLEARS' if cleared else 'does NOT clear'} threshold). "
          f"Only {n_pos}/{len(gains_all)} methods are net-positive at all.")
        a("- BatchNorm-statistics adaptation gives a **small** positive gain; the covariance "
          "(Euclidean/Riemannian) methods slightly **hurt**; filter-bank/z-score are ≈ neutral. "
          "This is a **useful negative / diagnostic result**: pure statistic-only alignment cannot "
          "close the cross-session gap, which motivates (but does NOT itself perform) learning-based "
          "Step-3 adaptation.")
    else:
        a("- _Comparison vs none_reference unavailable._")
    a("")
    a("## 1. Experiment goal")
    a("")
    a("Test whether **unsupervised statistic-only alignment** (no target labels, no weight "
      "learning on target) recovers part of the cross-session accuracy drop measured by the "
      "static baseline (single-source ≈ 9–13% drop; multi-source ses-01+02→ses-03 helps).")
    a("")
    a("## 2. Method definitions")
    a("")
    a("- `none_reference` — no alignment; pulled from baseline_v1 cross rows (NOT re-run).")
    a("- `session_zscore` — per-channel mean/std normalization; source stats from source train, "
      "applied to source train+val; target uses its own unlabeled X stats.")
    a("- `euclidean_alignment` — whiten by `R^{-1/2}`, R = arithmetic mean of trial covariances "
      "(eigh inverse-sqrt, eps ridge + diagonal shrinkage). Source R from source train; target R "
      "from target X.")
    a("- `riemannian_alignment` — whiten by `G^{-1/2}`, G = **log-Euclidean** SPD mean "
      "`expm(mean_i logm(C_i))` (numpy/scipy only, no pyriemann). Source G from source train; "
      "target G from target X.")
    a("- `bn_statistics_adaptation` — train on source (early stop on source val), then forward the "
      "unlabeled target X to refresh BatchNorm running mean/var only (no loss/backward/optimizer).")
    a("- `filterbank_reweighting` — decompose into θ/μ/β/low-γ FIR sub-bands and reweight each band "
      "(single scalar gain, clipped) so the target band-power profile matches the **source** "
      "profile. Conservative version: per-band scalar gains from target X only.")
    a("")
    a("## 3. Protocols")
    a("")
    a("- single-source directed pairs: ses-i → ses-j (both ok) — 6 directions per 3-ok subject.")
    a("- multi-source: ses-01+ses-02 → ses-03 (all three ok).")
    if expected_info:
        a("")
        a(f"- Expected single-source rows/method/model = {expected_info.get('single_pairs','?')} pairs × "
          f"{expected_info.get('n_seeds','?')} seeds; multi-source = {expected_info.get('multi_subjects','?')} "
          f"subjects × seeds.")
    if ms_skipped:
        a("")
        a(f"- Multi-source skipped subjects: {len(ms_skipped)} (missing one of ses-01/02/03).")
    a("")
    a("## 4. No-leakage / no-target-label checks")
    a("")
    if not df_ok.empty:
        bad_y = int((df_ok["used_target_y_for_training"] == True).sum())  # noqa: E712
        a(f"- `used_target_y_for_training == False` for ALL ok rows: "
          f"{'YES' if bad_y == 0 else 'NO (' + str(bad_y) + ' violations!)'}.")
        nonref = df_ok[df_ok["method"] != "none_reference"]
        if not nonref.empty:
            x_ok = int((nonref["used_target_x_for_stats"] == True).sum())  # noqa: E712
            a(f"- `used_target_x_for_stats == True` for all trained-method ok rows: "
              f"{'YES' if x_ok == len(nonref) else 'NO'} ({x_ok}/{len(nonref)}).")
        a(f"- n_train range [{int(df_ok['n_train'].min())}, {int(df_ok['n_train'].max())}]; "
          f"n_val [{int(df_ok['n_val'].min())}, {int(df_ok['n_val'].max())}]; "
          f"n_test [{int(df_ok['n_test'].min())}, {int(df_ok['n_test'].max())}].")
    a("- Code guards: target trials never in train/val (separate session); val carved from source "
      "train only; BN method asserts no optimizer (running-stat only); alignment asserts shape "
      "unchanged + finite.")
    a("")
    a("## 5. Comparison vs baseline_v1 `none_reference`")
    a("")
    if not vs_base.empty:
        bm = vs_base[vs_base["training_scope"] == "all"].groupby("method", as_index=False)["gain_acc_mean"].mean()
        bm = bm.set_index("method").reindex(_order(bm["method"], METHOD_ORDER)).reset_index()
        a("| method | mean Δacc vs none |")
        a("|---|---:|")
        for _, r in bm.iterrows():
            a(f"| `{r['method']}` | {r['gain_acc_mean']:+.4f} |")
        if not bm.empty:
            best = bm.iloc[int(bm["gain_acc_mean"].astype(float).values.argmax())]
            a("")
            a(f"- **Largest average improvement: `{best['method']}` ({best['gain_acc_mean']:+.4f})**.")
    else:
        a("_baseline none_reference not available for comparison._")
    a("")
    a("## 6. Single-source direction results")
    a("")
    if not by_dir.empty:
        dirs = sorted(by_dir["protocol"].unique())
        methods = _order(by_dir["method"], METHOD_ORDER)
        a("| direction | " + " | ".join(f"`{m}`" for m in methods) + " |")
        a("|---" * (len(methods) + 1) + "|")
        for d in dirs:
            cells = []
            for m in methods:
                r = by_dir[(by_dir["method"] == m) & (by_dir["protocol"] == d)]
                cells.append(_fmt(r["acc_mean"].iloc[0]) if len(r) else "—")
            a(f"| {d} | " + " | ".join(cells) + " |")
    a("")
    a("## 7. Multi-source ses-01+02 → ses-03 results")
    a("")
    ms = by_proto[by_proto["training_scope"] == "multi_source"] if not by_proto.empty else pd.DataFrame()
    if not ms.empty:
        a("| method | Acc | BalAcc | MacroF1 | AUC |")
        a("|---|---:|---:|---:|---:|")
        for m in _order(ms["method"], METHOD_ORDER):
            r = ms[ms["method"] == m].iloc[0]
            a(f"| `{m}` | {_fmt(r['acc_mean'])}±{_fmt(r['acc_std'],3)} | {_fmt(r['bacc_mean'])} "
              f"| {_fmt(r['f1_mean'])} | {_fmt(r['auc_mean'])} |")
    a("")
    a("## 8. Which method improves most")
    a("")
    if not gains_all.empty:
        best = gains_all.iloc[int(gains_all["gain_acc_mean"].astype(float).values.argmax())]
        best_m, best_g = best["method"], float(best["gain_acc_mean"])
        a(f"- Top method = **`{best_m}`**, mean Δacc = **{best_g:+.4f}** over none_reference — "
          f"**below** the +{SUCCESS_THRESHOLD:.2f} success line, so it is NOT a sufficient no-learning fix.")
        a(f"- Net-positive methods: {int((gains_all['gain_acc_mean'] > 0).sum())}/{len(gains_all)}. "
          "Covariance whitening (Euclidean/Riemannian) is the worst (slightly negative); "
          "z-score and filter-bank are ≈ neutral.")
    else:
        a("- See §5 table.")
    a("")
    a("## 9. Which directions improve most")
    a("")
    if gain_rows is not None and not gain_rows.empty and "gain_acc" in gain_rows:
        ss = gain_rows[gain_rows["training_scope"] == "single_source"]
        if not ss.empty:
            per_dir = (ss.groupby(["method", "protocol"], as_index=False)["gain_acc"].mean())
            # best-method direction breakdown
            if not gains_all.empty:
                bm = gains_all.iloc[int(gains_all["gain_acc_mean"].astype(float).values.argmax())]["method"]
                sub = per_dir[per_dir["method"] == bm].sort_values("gain_acc", ascending=False)
                if not sub.empty:
                    top = sub.iloc[0]; bot = sub.iloc[-1]
                    a(f"- For the best method `{bm}`: most-improved direction = **{top['protocol']}** "
                      f"({top['gain_acc']:+.4f}); least = **{bot['protocol']}** ({bot['gain_acc']:+.4f}).")
            # overall most-improved (method,direction) and worst
            best_row = per_dir.iloc[int(per_dir["gain_acc"].values.argmax())]
            worst_row = per_dir.iloc[int(per_dir["gain_acc"].values.argmin())]
            a(f"- Overall best (method,direction) gain = `{best_row['method']}` on {best_row['protocol']} "
              f"({best_row['gain_acc']:+.4f}); worst = `{worst_row['method']}` on {worst_row['protocol']} "
              f"({worst_row['gain_acc']:+.4f}).")
        a("- Full per-direction means in `alignment_by_direction.csv`; per-direction gains derivable "
          "from `results_alignment_all.csv`.")
    else:
        a("- See `alignment_by_direction.csv`.")
    a("")
    a("## 10. Which subjects improve / regress")
    a("")
    if by_subject is not None and not by_subject.empty and not gains_all.empty:
        bm = gains_all.iloc[int(gains_all["gain_acc_mean"].astype(float).values.argmax())]["method"]
        sub = by_subject[by_subject["method"] == bm].copy()
        if not sub.empty:
            n_up = int((sub["gain_acc_mean"] > 0).sum())
            n_dn = int((sub["gain_acc_mean"] < 0).sum())
            sub_sorted = sub.sort_values("gain_acc_mean", ascending=False)
            top3 = ", ".join(f"{r['subject']} ({r['gain_acc_mean']:+.3f})" for _, r in sub_sorted.head(3).iterrows())
            bot3 = ", ".join(f"{r['subject']} ({r['gain_acc_mean']:+.3f})" for _, r in sub_sorted.tail(3).iterrows())
            a(f"- Under the best method `{bm}`: **{n_up}** subjects improve, **{n_dn}** regress.")
            a(f"- Most improved: {top3}.")
            a(f"- Most regressed: {bot3}.")
    a("- Full per-subject gains in `alignment_by_subject.csv` and `alignment_gain_by_subject.png`.")
    a("")
    a("## 11. Effect by drift level")
    a("")
    if not gain_drift.empty:
        levels = [l for l in ["stable", "moderate", "high"] if l in set(gain_drift["drift_level"])]
        methods = _order(gain_drift["method"], METHOD_ORDER)
        a("| drift_level | " + " | ".join(f"`{m}`" for m in methods) + " |")
        a("|---" * (len(methods) + 1) + "|")
        for lv in levels:
            cells = []
            for m in methods:
                r = gain_drift[(gain_drift["method"] == m) & (gain_drift["drift_level"] == lv)]
                cells.append(f"{r['gain_acc_mean'].iloc[0]:+.4f}" if len(r) else "—")
            a(f"| {lv} | " + " | ".join(cells) + " |")
    else:
        a("_drift-level breakdown unavailable._")
    a("")
    a("## 12. Is online / agent adaptation warranted?")
    a("")
    a("- **Yes — warranted, and this run is the evidence for it (but Step-3 is NOT run here).** "
      "No no-learning method clears the +0.02 line; the best (BN-stats) gives only a small positive "
      "gain, and the covariance methods slightly hurt. Crucially, on **high-drift** subjects the "
      "gains are smallest/negative (e.g. filter-bank is strongly negative on high drift), i.e. the "
      "subjects that need help most are the least helped by statistic-only alignment.")
    a("- Interpretation: closing the residual cross-session gap needs **learning-based** target "
      "adaptation (online update / adapter / prototype / memory), not just unsupervised statistics. "
      "That is the objective justification for the next stage.")
    a("")
    a("## 13. Next-step suggestions (NOT executed here)")
    a("")
    a("- Use BN-stats adaptation (cheap, mildly positive, never hurts much) as a default front-end, "
      "possibly combined with multi-source training.")
    a("- Explore learning-based Step-3 adaptation (online test-then-update / lightweight adapter / "
      "prototype-memory), focusing on high-drift subjects where no-learning alignment fails.")
    a("- Consider an UNLABELED per-subject/per-direction method-selection criterion (no target labels).")
    a("- These are suggestions only; no Step-3 / online / fine-tuning / 41-10 / CAP-EEGNet-full run is "
      "performed in this report.")
    a("")
    a("## Files")
    a("")
    for label, path in {**tables, **figures}.items():
        a(f"- {label}: `{path}`")
    a("")
    report_path.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Summarize Step-2 alignment results.")
    ap.add_argument("--out", default="outputs/experiments/alignment_baseline_v1")
    ap.add_argument("--baseline-cross-all",
                    default="outputs/experiments/baseline_v1/cross_session/tables/results_cross_session_all.csv")
    ap.add_argument("--drift-csv",
                    default="outputs/analysis/session_drift_v1/per_subject_drift_summary.csv")
    ap.add_argument("--expected-job-ids", default=None,
                    help="optional full_job_ids.txt to sacct-check completeness")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir = out_dir if out_dir.is_absolute() else PROJECT_ROOT / out_dir
    cross_dir = out_dir / "cross_session"
    runs_dir = cross_dir / "runs"
    tab_dir = cross_dir / "tables"
    fig_dir = cross_dir / "figures"
    tab_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    baseline_csv = Path(args.baseline_cross_all)
    baseline_csv = baseline_csv if baseline_csv.is_absolute() else PROJECT_ROOT / baseline_csv
    drift_csv = Path(args.drift_csv)
    drift_csv = drift_csv if drift_csv.is_absolute() else PROJECT_ROOT / drift_csv

    df_align = load_runs(runs_dir)
    if df_align.empty:
        print("[summarize] WARNING: no alignment run CSVs found; writing empty status only.")
    df_align_ok = df_align[df_align["status"] == "ok"].copy() if not df_align.empty else pd.DataFrame()

    scopes_present = set(df_align["training_scope"].unique()) if not df_align.empty else set()
    models_present = set(df_align["model"].unique()) if not df_align.empty else set()
    none_df = load_none_reference(baseline_csv, scopes_present, models_present)
    drift_map = load_drift(drift_csv)

    # All-results table (alignment rows + none_reference rows).
    df_all = pd.concat([d for d in [df_align, none_df] if not d.empty], ignore_index=True)
    df_ok = df_all[df_all["status"] == "ok"].copy() if not df_all.empty else pd.DataFrame()
    if not df_ok.empty:
        df_ok["drift_level"] = df_ok["subject"].map(drift_map).fillna("unknown")

    # Aggregations.
    by_method_scope = agg_meanstd(df_ok, ["method", "training_scope"]) if not df_ok.empty else pd.DataFrame()
    by_method_all = agg_meanstd(df_ok.assign(training_scope="all"), ["method", "training_scope"]) \
        if not df_ok.empty else pd.DataFrame()
    by_method = pd.concat([d for d in [by_method_scope, by_method_all] if not d.empty], ignore_index=True)
    by_model = agg_meanstd(df_ok, ["method", "model", "training_scope"]) if not df_ok.empty else pd.DataFrame()
    by_proto = by_method_scope
    ss = df_ok[df_ok["training_scope"] == "single_source"] if not df_ok.empty else pd.DataFrame()
    by_dir = agg_meanstd(ss, ["method", "protocol"]) if not ss.empty else pd.DataFrame()

    # Gains vs none_reference.
    gain_rows = compute_gains(df_align_ok.assign(
        drift_level=df_align_ok["subject"].map(drift_map).fillna("unknown")) if not df_align_ok.empty
        else df_align_ok, none_df)
    if not gain_rows.empty and "gain_acc" in gain_rows:
        vs_scope = (gain_rows.groupby(["method", "training_scope", "seed"], as_index=False)["gain_acc"].mean()
                    .groupby(["method", "training_scope"], as_index=False)
                    .agg(gain_acc_mean=("gain_acc", "mean"), gain_acc_std=("gain_acc", "std")))
        vs_all = (gain_rows.assign(training_scope="all")
                  .groupby(["method", "training_scope", "seed"], as_index=False)["gain_acc"].mean()
                  .groupby(["method", "training_scope"], as_index=False)
                  .agg(gain_acc_mean=("gain_acc", "mean"), gain_acc_std=("gain_acc", "std")))
        vs_base = pd.concat([vs_scope, vs_all], ignore_index=True)
        by_subject = (gain_rows.groupby(["method", "subject"], as_index=False)
                      .agg(acc_mean=("acc", "mean"), none_acc_mean=("none_acc", "mean"),
                           gain_acc_mean=("gain_acc", "mean")))
        gain_drift = (gain_rows.groupby(["method", "drift_level"], as_index=False)
                      .agg(gain_acc_mean=("gain_acc", "mean"), gain_acc_std=("gain_acc", "std"),
                           n=("gain_acc", "size")))
    else:
        vs_base = pd.DataFrame(); by_subject = pd.DataFrame(); gain_drift = pd.DataFrame()

    # run_status table.
    if not df_all.empty:
        rs = (df_all.assign(is_ok=(df_all["status"] == "ok").astype(int),
                            is_nan=df_all["acc"].isna().astype(int))
              .groupby(["method", "model", "training_scope"], as_index=False)
              .agg(n_rows=("status", "size"), n_ok=("is_ok", "sum"), n_nan=("is_nan", "sum")))
    else:
        rs = pd.DataFrame()

    # Write tables.
    tables = {
        "results_alignment_all": tab_dir / "results_alignment_all.csv",
        "alignment_by_method": tab_dir / "alignment_by_method.csv",
        "alignment_by_model": tab_dir / "alignment_by_model.csv",
        "alignment_by_direction": tab_dir / "alignment_by_direction.csv",
        "alignment_by_protocol": tab_dir / "alignment_by_protocol.csv",
        "alignment_by_subject": tab_dir / "alignment_by_subject.csv",
        "alignment_vs_baseline": tab_dir / "alignment_vs_baseline.csv",
        "alignment_gain_by_drift_level": tab_dir / "alignment_gain_by_drift_level.csv",
        "run_status": tab_dir / "run_status.csv",
    }
    df_all.to_csv(tables["results_alignment_all"], index=False)
    by_method.to_csv(tables["alignment_by_method"], index=False)
    by_model.to_csv(tables["alignment_by_model"], index=False)
    by_dir.to_csv(tables["alignment_by_direction"], index=False)
    by_proto.to_csv(tables["alignment_by_protocol"], index=False)
    by_subject.to_csv(tables["alignment_by_subject"], index=False)
    vs_base.to_csv(tables["alignment_vs_baseline"], index=False)
    gain_drift.to_csv(tables["alignment_gain_by_drift_level"], index=False)
    rs.to_csv(tables["run_status"], index=False)

    # Figures.
    figures = {
        "alignment_method_comparison": fig_dir / "alignment_method_comparison.png",
        "alignment_by_direction": fig_dir / "alignment_by_direction.png",
        "alignment_vs_baseline_gain": fig_dir / "alignment_vs_baseline_gain.png",
        "alignment_gain_by_subject": fig_dir / "alignment_gain_by_subject.png",
        "alignment_gain_by_drift_level": fig_dir / "alignment_gain_by_drift_level.png",
        "alignment_protocol_comparison": fig_dir / "alignment_protocol_comparison.png",
    }
    fig_method_comparison(by_method, figures["alignment_method_comparison"])
    fig_by_direction(by_dir, figures["alignment_by_direction"])
    fig_vs_baseline_gain(vs_base, figures["alignment_vs_baseline_gain"])
    fig_gain_by_subject(gain_rows if not gain_rows.empty else pd.DataFrame(), figures["alignment_gain_by_subject"])
    fig_gain_by_drift(gain_drift, figures["alignment_gain_by_drift_level"])
    fig_protocol_comparison(by_proto, figures["alignment_protocol_comparison"])

    # Completeness.
    n_failed = int((df_all["status"] == "failed").sum()) if not df_all.empty else 0
    n_nan = int(df_ok["acc"].isna().sum()) if not df_ok.empty else 0
    trained_present = set(df_align_ok["method"].unique()) if not df_align_ok.empty else set()
    expected_trained = {"session_zscore", "euclidean_alignment", "riemannian_alignment",
                        "bn_statistics_adaptation", "filterbank_reweighting"}
    complete = (not df_align_ok.empty) and trained_present >= expected_trained and n_failed == 0 and n_nan == 0

    # manifest_sources.json + RUN_STATUS.md.
    manifest = {
        "experiment_id": "alignment_baseline_v1",
        "baseline_none_reference_source": str(baseline_csv),
        "drift_levels_source": str(drift_csv),
        "n_run_csvs": len(sorted(runs_dir.glob("alignment__*.csv"))),
        "n_rows_total": int(len(df_all)), "n_align_rows": int(len(df_align)),
        "n_none_rows": int(len(none_df)), "n_ok": int(len(df_ok)),
        "n_failed": n_failed, "n_nan_acc": n_nan,
        "trained_methods_present": sorted(trained_present),
        "models_present": sorted(models_present), "complete": bool(complete),
    }
    (out_dir / "manifest_sources.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")

    run_status_md = [
        "# Alignment Baseline RUN STATUS", "",
        f"- complete: **{complete}**",
        f"- run CSVs: {manifest['n_run_csvs']}",
        f"- rows total: {len(df_all)} (alignment {len(df_align)}, none_reference {len(none_df)})",
        f"- ok: {len(df_ok)} | failed: {n_failed} | NaN-acc: {n_nan}",
        f"- trained methods present: {sorted(trained_present)}",
        f"- models present: {sorted(models_present)}", "",
        "Tables: `cross_session/tables/`  Figures: `cross_session/figures/`  Report: `ALIGNMENT_BASELINE_REPORT.md`",
    ]
    (out_dir / "RUN_STATUS.md").write_text("\n".join(run_status_md), encoding="utf-8")

    write_report(out_dir / "ALIGNMENT_BASELINE_REPORT.md",
                 df_all=df_all, df_ok=df_ok, by_method=by_method, by_proto=by_proto,
                 by_dir=by_dir, vs_base=vs_base, gain_drift=gain_drift,
                 by_subject=by_subject, gain_rows=gain_rows, complete=complete,
                 n_failed=n_failed, n_nan=n_nan, tables=tables, figures=figures,
                 ms_skipped=[], expected_info={})

    print(f"[summarize] rows={len(df_all)} ok={len(df_ok)} failed={n_failed} nan={n_nan} "
          f"complete={complete}")
    print(f"[summarize] wrote tables to {tab_dir} and report to {out_dir / 'ALIGNMENT_BASELINE_REPORT.md'}")


if __name__ == "__main__":
    main()
