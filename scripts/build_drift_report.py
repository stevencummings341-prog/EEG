#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build per-pair / per-subject drift tables + figures + report from an existing drift CSV.

Dataset/session-agnostic generalization of the legacy WBCIC-only builder. It reads the
``session_drift_report.csv`` produced by ``code/run.py --config <phaseN_drift>`` (one row
per within-subject directed/undirected session pair) and derives EVERYTHING from the data
(session count, pair set, partial subjects) instead of hardcoding 3 sessions / fixed
failed sessions. Works for WBCIC (3 ses) and SHU (5 ses) alike. It does NOT recompute any
drift metric (never touches npz) and does not submit jobs.

Inputs (read-only):
  <dir>/session_drift_report.csv

New outputs (added next to the CSV, originals untouched):
  session_pair_summary.csv            per pair-type aggregate
  per_subject_drift_summary.csv/.md   per-subject drift profile + drift_score + drift_level
  figures/session_pair_metric_summary.png
  figures/subject_mmd_heatmap.png
  figures/subject_csp_heatmap.png
  figures/subject_erd_mu_heatmap.png
  figures/high_drift_subjects_bar.png
  figures/signal_quality_shift.png
  DRIFT_PAIR_SUBJECT_REPORT.md        per-pair + per-subject narrative report

Run (light, CPU):
  python scripts/build_drift_report.py --dir outputs/analysis/shu/session_drift_v1 \
      --dataset shu

依赖: numpy >= 1.21, pandas >= 1.3, matplotlib >= 3.4
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ses_num(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"(\d+)")[0]


def load_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["pair"] = _ses_num(df["ses_i"]) + "-" + _ses_num(df["ses_j"])
    return df


def pair_order(df: pd.DataFrame) -> List[str]:
    """All pair labels present, sorted by (i, j) numerically."""
    pairs = sorted(set(df["pair"]), key=lambda p: tuple(int(x) for x in p.split("-")))
    return pairs


def _fmt(v, nd=3):
    return "—" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.{nd}f}"


def overall_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    cols = ["mmd", "coral", "mu_power_shift", "beta_power_shift", "mu_ks_stat",
            "erd_mu_corr", "erd_beta_corr", "csp_similarity", "rms_ratio_median",
            "fisher_i", "fisher_j", "fisher_shift"]
    out = {}
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c].dropna()
        out[c] = {"mean": float(s.mean()), "median": float(s.median()),
                  "std": float(s.std(ddof=0)), "n": int(len(s))}
    return out


def session_pair_summary(df: pd.DataFrame, pairs: List[str]) -> pd.DataFrame:
    rows = []
    for pair in pairs:
        g = df[df["pair"] == pair]
        if not len(g):
            continue
        rows.append({
            "pair": pair, "n": int(len(g)),
            "mmd_mean": g["mmd"].mean(), "mmd_median": g["mmd"].median(),
            "mu_ks_mean": g["mu_ks_stat"].mean(),
            "mu_power_shift_mean": g["mu_power_shift"].mean(),
            "beta_power_shift_mean": g["beta_power_shift"].mean(),
            "csp_similarity_mean": g["csp_similarity"].mean(),
            "erd_mu_corr_mean": g["erd_mu_corr"].mean(),
            "erd_beta_corr_mean": g["erd_beta_corr"].mean(),
            "rms_ratio_median": g["rms_ratio_median"].median(),
            "fisher_shift_mean": g["fisher_shift"].mean(),
        })
    return pd.DataFrame(rows)


def per_subject_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subj, g in df.groupby("subject"):
        rows.append({
            "subject": subj,
            "available_pairs": ",".join(sorted(g["pair"].tolist())),
            "n_pairs": int(len(g)),
            "mean_mmd": g["mmd"].mean(), "max_mmd": g["mmd"].max(),
            "mean_mu_ks": g["mu_ks_stat"].mean(),
            "mean_csp_similarity": g["csp_similarity"].mean(),
            "mean_erd_mu_corr": g["erd_mu_corr"].mean(),
            "mean_erd_beta_corr": g["erd_beta_corr"].mean(),
            "median_rms_ratio": g["rms_ratio_median"].median(),
            "mean_abs_fisher_shift": g["fisher_shift"].abs().mean(),
        })
    sub = pd.DataFrame(rows)

    # Composite drift score: higher = more drift. z-standardize across subjects;
    # distance metrics add (+), stability metrics (csp/erd) subtract (−).
    def z(col, sign):
        v = sub[col].astype(float)
        sd = v.std(ddof=0)
        zz = (v - v.mean()) / sd if sd > 1e-12 else v * 0.0
        return sign * zz
    sub["drift_score"] = (z("mean_mmd", +1) + z("mean_mu_ks", +1)
                          + z("mean_csp_similarity", -1)
                          + z("mean_erd_mu_corr", -1) + z("mean_erd_beta_corr", -1)) / 5.0
    q1, q2 = sub["drift_score"].quantile([1 / 3, 2 / 3])

    def level(s):
        return "high" if s >= q2 else ("stable" if s <= q1 else "moderate")
    sub["drift_level"] = sub["drift_score"].apply(level)
    return sub.sort_values("drift_score", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_pair_metric_summary(df, pairs, out: Path) -> None:
    metrics = [("mmd", "MMD"), ("csp_similarity", "CSP similarity"),
               ("erd_mu_corr", "ERD/ERS mu corr"), ("mu_ks_stat", "mu KS stat")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, (col, title) in zip(axes, metrics):
        data = [df.loc[df["pair"] == p, col].dropna().values for p in pairs]
        ax.boxplot(data, showmeans=True)
        ax.set_xticks(range(1, len(pairs) + 1)); ax.set_xticklabels(pairs, rotation=45)
        ax.set_title(title); ax.set_xlabel("session pair")
    fig.suptitle("Cross-session drift by session pair")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def _subject_pair_matrix(df, pairs, col: str):
    subs = sorted(df["subject"].unique())
    mat = np.full((len(subs), len(pairs)), np.nan)
    sidx = {s: i for i, s in enumerate(subs)}
    pidx = {p: j for j, p in enumerate(pairs)}
    for _, r in df.iterrows():
        if r["pair"] in pidx:
            mat[sidx[r["subject"]], pidx[r["pair"]]] = r[col]
    return subs, mat


def fig_subject_heatmap(df, pairs, col, title, cmap, out: Path, vmin=None, vmax=None) -> None:
    subs, mat = _subject_pair_matrix(df, pairs, col)
    fig, ax = plt.subplots(figsize=(max(5.5, len(pairs) * 0.8), max(8, len(subs) * 0.22)))
    cm = plt.get_cmap(cmap).copy(); cm.set_bad("lightgray")
    im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", cmap=cm, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pairs))); ax.set_xticklabels(pairs, rotation=45)
    ax.set_yticks(range(len(subs)))
    ax.set_yticklabels([s.replace("sub-", "") for s in subs], fontsize=6)
    ax.set_xlabel("session pair"); ax.set_ylabel("subject"); ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_high_drift_bar(sub, out: Path, top_n: int = 10) -> None:
    top = sub.head(top_n)
    x = np.arange(len(top)); w = 0.27
    fig, ax = plt.subplots(figsize=(1.1 * len(top) + 3, 5))
    ax.bar(x - w, top["mean_mmd"], w, label="mean MMD")
    ax.bar(x, top["mean_mu_ks"], w, label="mean mu KS")
    ax.bar(x + w, top["mean_csp_similarity"], w, label="mean CSP sim")
    ax.set_xticks(x); ax.set_xticklabels([s.replace("sub-", "") for s in top["subject"]],
                                         rotation=45, ha="right")
    ax.set_xlabel("subject (top-%d by drift score)" % top_n)
    ax.set_title("Top high-drift subjects"); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_signal_quality(df, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    hi_shift = (df["high_amp_ratio_j"] - df["high_amp_ratio_i"]).dropna()
    axes[0].hist(hi_shift, bins=25, edgecolor="black", alpha=0.75)
    axes[0].axvline(0, color="red", linestyle="--", label="no change")
    axes[0].set_title("High-amplitude trial ratio shift (j - i)")
    axes[0].set_xlabel("delta high-amp trial fraction"); axes[0].set_ylabel("session pairs")
    axes[0].legend()
    rms_ratio = (df["mean_rms_j"] / df["mean_rms_i"]).replace([np.inf, -np.inf], np.nan).dropna()
    axes[1].hist(rms_ratio, bins=25, edgecolor="black", alpha=0.75)
    axes[1].axvline(1.0, color="red", linestyle="--", label="no change")
    axes[1].set_title("Mean-RMS ratio (session j / session i)")
    axes[1].set_xlabel("mean RMS ratio"); axes[1].set_ylabel("session pairs"); axes[1].legend()
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


# --------------------------------------------------------------------------- #
# Markdown writers
# --------------------------------------------------------------------------- #
def write_per_subject_md(sub: pd.DataFrame, path: Path) -> None:
    L: List[str] = ["# Per-subject cross-session drift summary\n"]
    L.append(f"{len(sub)} subjects (>=2 ok sessions to form a pair). `drift_score` higher = "
             "more drift (= mean of z(MMD)+z(mu_KS)-z(CSP)-z(ERD_mu)-z(ERD_beta)); "
             "tertile split into high / moderate / stable.\n")
    L.append("| subject | available_pairs | n_pairs | mean_mmd | max_mmd | mean_mu_ks | "
             "mean_csp_sim | mean_erd_mu | mean_erd_beta | median_rms_ratio | "
             "mean_abs_fisher_shift | drift_score | drift_level |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sub.itertuples():
        L.append(f"| {r.subject} | {r.available_pairs} | {r.n_pairs} | {_fmt(r.mean_mmd)} | "
                 f"{_fmt(r.max_mmd)} | {_fmt(r.mean_mu_ks)} | {_fmt(r.mean_csp_similarity)} | "
                 f"{_fmt(r.mean_erd_mu_corr)} | {_fmt(r.mean_erd_beta_corr)} | "
                 f"{_fmt(r.median_rms_ratio)} | {_fmt(r.mean_abs_fisher_shift)} | "
                 f"{_fmt(r.drift_score)} | {r.drift_level} |")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def write_report(df, dataset, stats, pair_df, sub, figures, path: Path) -> None:
    m = stats
    n_subj = df["subject"].nunique()
    n_pairs = len(df)
    n_full = int((sub["n_pairs"] == sub["n_pairs"].max()).sum())
    high = sub[sub["drift_level"] == "high"]; stable = sub[sub["drift_level"] == "stable"]
    worst_pair = pair_df.loc[pair_df["mmd_mean"].idxmax(), "pair"] if len(pair_df) else "—"

    L: List[str] = []
    L.append(f"# Cross-session drift: per-pair / per-subject report ({dataset})\n")
    L.append(f"> Built from `session_drift_report.csv` ({n_pairs} within-subject session "
             f"pairs / {n_subj} subjects). No drift metric recomputed.\n")
    L.append("## A. Goal\n")
    L.append("Quantify within-subject EEG distribution drift across sessions (different days). "
             "Data-level diagnostic only; not model training.\n")
    L.append("## B. Overall stats\n")
    L.append("| metric | mean | median | std |")
    L.append("|---|---|---|---|")
    for c in ["mmd", "coral", "mu_power_shift", "beta_power_shift", "mu_ks_stat",
              "erd_mu_corr", "erd_beta_corr", "csp_similarity", "rms_ratio_median", "fisher_shift"]:
        if c in m:
            L.append(f"| `{c}` | {_fmt(m[c]['mean'],4)} | {_fmt(m[c]['median'],4)} | {_fmt(m[c]['std'],4)} |")
    L.append("")
    L.append("## C. By session pair\n")
    L.append("| pair | n | MMD mean | MMD median | mu-KS mean | CSP sim mean | ERD mu corr | "
             "ERD beta corr | RMS ratio median | Fisher shift mean |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in pair_df.itertuples():
        L.append(f"| **{r.pair}** | {r.n} | {_fmt(r.mmd_mean)} | {_fmt(r.mmd_median)} | "
                 f"{_fmt(r.mu_ks_mean)} | {_fmt(r.csp_similarity_mean)} | {_fmt(r.erd_mu_corr_mean)} | "
                 f"{_fmt(r.erd_beta_corr_mean)} | {_fmt(r.rms_ratio_median)} | {_fmt(r.fisher_shift_mean,4)} |")
    L.append("")
    L.append(f"- MMD-largest pair: `{worst_pair}`.\n")
    L.append("## D. By subject\n")
    L.append(f"{n_full} subjects have the full pair set; high-drift {len(high)}, stable {len(stable)}. "
             "Full table: `per_subject_drift_summary.csv` / `.md`.\n")
    L.append("### Top 10 high-drift subjects\n")
    L.append("| subject | pairs | mean_mmd | mean_mu_ks | mean_csp_sim | mean_erd_mu | drift_score |")
    L.append("|---|---|---|---|---|---|---|")
    for r in sub.head(10).itertuples():
        L.append(f"| {r.subject} | {r.available_pairs} | {_fmt(r.mean_mmd)} | {_fmt(r.mean_mu_ks)} | "
                 f"{_fmt(r.mean_csp_similarity)} | {_fmt(r.mean_erd_mu_corr)} | {_fmt(r.drift_score)} |")
    L.append("\n### Top 10 stable subjects\n")
    L.append("| subject | pairs | mean_mmd | mean_mu_ks | mean_csp_sim | mean_erd_mu | drift_score |")
    L.append("|---|---|---|---|---|---|---|")
    for r in sub.tail(10).iloc[::-1].itertuples():
        L.append(f"| {r.subject} | {r.available_pairs} | {_fmt(r.mean_mmd)} | {_fmt(r.mean_mu_ks)} | "
                 f"{_fmt(r.mean_csp_similarity)} | {_fmt(r.mean_erd_mu_corr)} | {_fmt(r.drift_score)} |")
    L.append("\n## E. Figures\n")
    for f in figures:
        L.append(f"- `figures/{f}`")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build drift per-pair/per-subject tables+figures+report.")
    ap.add_argument("--dir", required=True, help="drift output dir containing session_drift_report.csv")
    ap.add_argument("--dataset", default="dataset", help="dataset label for the report title")
    args = ap.parse_args()
    d = Path(args.dir)
    csv_path = d / "session_drift_report.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing {csv_path} (run the drift diagnostic first).")
    fig_dir = d / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_df(csv_path)
    pairs = pair_order(df)
    stats = overall_stats(df)
    pair_df = session_pair_summary(df, pairs)
    sub = per_subject_summary(df)

    pair_df.to_csv(d / "session_pair_summary.csv", index=False, float_format="%.6f")
    sub.to_csv(d / "per_subject_drift_summary.csv", index=False, float_format="%.6f")
    write_per_subject_md(sub, d / "per_subject_drift_summary.md")

    figures: List[str] = []
    fig_pair_metric_summary(df, pairs, fig_dir / "session_pair_metric_summary.png")
    figures.append("session_pair_metric_summary.png")
    fig_subject_heatmap(df, pairs, "mmd", "MMD by subject x session-pair", "YlOrRd",
                        fig_dir / "subject_mmd_heatmap.png")
    figures.append("subject_mmd_heatmap.png")
    fig_subject_heatmap(df, pairs, "csp_similarity", "CSP similarity by subject x pair", "YlGnBu",
                        fig_dir / "subject_csp_heatmap.png", vmin=0.0, vmax=1.0)
    figures.append("subject_csp_heatmap.png")
    fig_subject_heatmap(df, pairs, "erd_mu_corr", "ERD/ERS mu corr by subject x pair", "YlGnBu",
                        fig_dir / "subject_erd_mu_heatmap.png", vmin=-1.0, vmax=1.0)
    figures.append("subject_erd_mu_heatmap.png")
    fig_high_drift_bar(sub, fig_dir / "high_drift_subjects_bar.png")
    figures.append("high_drift_subjects_bar.png")
    fig_signal_quality(df, fig_dir / "signal_quality_shift.png")
    figures.append("signal_quality_shift.png")
    existing = ["distribution_distance_hist.png", "band_power_shift_hist.png",
                "erd_ers_correlation_hist.png", "csp_similarity_hist.png",
                "fisher_ratio_scatter.png", "rms_ratio_hist.png",
                "metric_correlation_matrix.png", "session_pair_comparison.png"]
    figures += [f for f in existing if (fig_dir / f).exists()]

    write_report(df, args.dataset, stats, pair_df, sub, figures, d / "DRIFT_PAIR_SUBJECT_REPORT.md")

    print(f"[drift-report] {len(df)} pairs / {df['subject'].nunique()} subjects | pairs={pairs}")
    print(f"[drift-report] high={int((sub['drift_level']=='high').sum())} "
          f"moderate={int((sub['drift_level']=='moderate').sum())} "
          f"stable={int((sub['drift_level']=='stable').sum())}")
    print(f"[drift-report] wrote report + 2 tables + {len(figures)} figures to {d}")


if __name__ == "__main__":
    main()
