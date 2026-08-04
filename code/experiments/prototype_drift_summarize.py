#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 2c prototype-drift summarizer (CPU; runs as a Slurm afterany dependency).

Merges all per-run CSVs under <run_dir>/runs/, computes correlation tables and
figures, writes an honest report, and syncs the readable layer into
4_experiments/prototype_drift/{tables,figures,report}. Always emits run_status.csv
and RUN_STATUS.md even if some GPU jobs failed -- failures are listed, never faked.

依赖: pandas, numpy, scipy, matplotlib (Agg, no seaborn).
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODELS = ["eegnet", "deepconvnet", "fbcnet"]
RELATIONSHIPS = [
    ("prototype_drift_mean", "acc_drop"),
    ("prototype_direction_cosine", "acc_drop"),
    ("separation_change", "acc_drop"),
    ("target_negative_margin_rate", "acc_drop"),
    ("target_margin_mean", "acc_drop"),
    ("fisher_change", "acc_drop"),
]
CANON_PTYPE = "label_based"
CANON_DIST = "euclidean"


def _abs(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


# --------------------------------------------------------------------------- #
# Merge per-run CSVs
# --------------------------------------------------------------------------- #
def _concat_glob(runs_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(runs_dir / pattern)))
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                frames.append(df)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] could not read {f}: {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _expected_cells(manifest_path: str, status_filter: Tuple[str, ...]) -> pd.DataFrame:
    """Expected (subject, source, target) directed cells (subjects with >=2 ok)."""
    from code.datasets.session_splits import load_ok_sessions
    from code.experiments.prototype_drift import expected_cells
    records = load_ok_sessions(manifest_path, status_filter=status_filter)
    rows = [{"subject": s, "source_session": src, "target_session": tgt,
             "direction": f"{src}->{tgt}"} for s, src, tgt in expected_cells(records)]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Correlation analysis
# --------------------------------------------------------------------------- #
def _stats(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    from scipy import stats
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = int(len(x))
    out = {"n": n, "pearson_r": np.nan, "pearson_p": np.nan,
           "spearman_rho": np.nan, "spearman_p": np.nan,
           "slope": np.nan, "intercept": np.nan, "r2": np.nan}
    if n < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return out
    pr = stats.pearsonr(x, y)
    sr = stats.spearmanr(x, y)
    lr = stats.linregress(x, y)
    out.update({"pearson_r": float(pr[0]), "pearson_p": float(pr[1]),
                "spearman_rho": float(sr[0]), "spearman_p": float(sr[1]),
                "slope": float(lr.slope), "intercept": float(lr.intercept),
                "r2": float(lr.rvalue ** 2)})
    return out


def compute_correlations(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    ok = metrics[metrics["status"] == "ok"].copy()
    for x_metric, y_metric in RELATIONSHIPS:
        for ptype in sorted(ok["prototype_type"].dropna().unique()):
            for dist in sorted(ok["distance"].dropna().unique()):
                sub = ok[(ok["prototype_type"] == ptype) & (ok["distance"] == dist)]
                model_groups = [(m, sub[sub["model"] == m]) for m in sorted(sub["model"].unique())]
                model_groups.append(("ALL", sub))
                for model, g in model_groups:
                    s = _stats(g[x_metric].to_numpy(dtype=float),
                               g[y_metric].to_numpy(dtype=float))
                    rows.append({"relationship": f"{x_metric}_vs_{y_metric}",
                                 "x_metric": x_metric, "y_metric": y_metric,
                                 "model": model, "prototype_type": ptype, "distance": dist,
                                 **s})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Run status
# --------------------------------------------------------------------------- #
def build_run_status(status: pd.DataFrame, expected: pd.DataFrame,
                     models: List[str], seeds: List[int]) -> pd.DataFrame:
    """Cross expected (subject,direction) x model x seed against observed status."""
    rows: List[Dict[str, object]] = []
    obs = {}
    if len(status) > 0:
        for _, r in status.iterrows():
            obs[(str(r["model"]), int(r["seed"]), str(r["subject"]), str(r["direction"]))] = str(r["status"])
    for _, c in expected.iterrows():
        for m in models:
            for sd in seeds:
                key = (m, int(sd), str(c["subject"]), str(c["direction"]))
                st = obs.get(key, "missing")
                rows.append({"model": m, "seed": int(sd), "subject": c["subject"],
                             "source_session": c["source_session"],
                             "target_session": c["target_session"],
                             "direction": c["direction"], "status": st})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _scatter_fig(df: pd.DataFrame, x: str, y: str, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    colors = {"eegnet": "#1f77b4", "deepconvnet": "#ff7f0e", "fbcnet": "#2ca02c"}
    for m in sorted(df["model"].unique()):
        g = df[df["model"] == m]
        ax.scatter(g[x], g[y], s=14, alpha=0.55, label=m,
                   color=colors.get(m, None), edgecolors="none")
    xv = df[x].to_numpy(dtype=float)
    yv = df[y].to_numpy(dtype=float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    if mask.sum() >= 3 and np.std(xv[mask]) > 1e-12:
        from scipy import stats
        lr = stats.linregress(xv[mask], yv[mask])
        rho = stats.spearmanr(xv[mask], yv[mask])[0]
        xs = np.linspace(xv[mask].min(), xv[mask].max(), 50)
        ax.plot(xs, lr.intercept + lr.slope * xs, "k--", lw=1.2,
                label=f"fit r2={lr.rvalue**2:.2f}, rho={rho:.2f}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def make_figures(metrics: pd.DataFrame, corr: pd.DataFrame, fig_dir: Path) -> List[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    made: List[str] = []
    ok = metrics[metrics["status"] == "ok"].copy()
    canon = ok[(ok["prototype_type"] == CANON_PTYPE) & (ok["distance"] == CANON_DIST)]
    if len(canon) == 0:  # fall back to whatever exists
        canon = ok

    scatter_specs = [
        ("prototype_drift_mean", "acc_drop", "drift_vs_acc_drop.png",
         "Prototype drift (mean) vs cross-session accuracy drop"),
        ("prototype_direction_cosine", "acc_drop", "direction_cosine_vs_acc_drop.png",
         "Prototype direction cosine vs accuracy drop"),
        ("target_negative_margin_rate", "acc_drop", "negative_margin_vs_acc_drop.png",
         "Target negative-margin rate vs accuracy drop"),
        ("separation_change", "acc_drop", "separation_change_vs_acc_drop.png",
         "Class separation change (source-target) vs accuracy drop"),
        ("fisher_change", "acc_drop", "fisher_change_vs_acc_drop.png",
         "Fisher ratio change (source-target) vs accuracy drop"),
    ]
    for x, y, fname, title in scatter_specs:
        if x in canon.columns and len(canon) > 0:
            _scatter_fig(canon, x, y, f"{title}\n(prototype={CANON_PTYPE}, distance={CANON_DIST})",
                         fig_dir / fname)
            made.append(fname)

    # acc_drop_by_model
    if len(ok) > 0:
        per_cell = ok.drop_duplicates(subset=["model", "seed", "subject", "direction"])
        means = per_cell.groupby("model")["acc_drop"].mean()
        stds = per_cell.groupby("model")["acc_drop"].std()
        fig, ax = plt.subplots(figsize=(6.0, 4.4))
        ax.bar(means.index, means.values, yerr=stds.reindex(means.index).values,
               capsize=4, color=["#1f77b4", "#ff7f0e", "#2ca02c"][:len(means)])
        ax.set_ylabel("mean acc_drop (source_val - target)")
        ax.set_title("Cross-session accuracy drop by model")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "acc_drop_by_model.png", dpi=130)
        plt.close(fig)
        made.append("acc_drop_by_model.png")

    # correlation_summary: Spearman rho per relationship (ALL-model, canonical subset)
    csub = corr[(corr["model"] == "ALL") & (corr["prototype_type"] == CANON_PTYPE)
                & (corr["distance"] == CANON_DIST)]
    if len(csub) > 0:
        fig, ax = plt.subplots(figsize=(8.0, 4.6))
        rels = csub["relationship"].tolist()
        rhos = csub["spearman_rho"].to_numpy(dtype=float)
        ax.barh(range(len(rels)), rhos, color="#4c72b0")
        ax.set_yticks(range(len(rels)))
        ax.set_yticklabels(rels, fontsize=8)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("Spearman rho vs acc_drop (ALL models)")
        ax.set_title(f"Correlation summary (prototype={CANON_PTYPE}, distance={CANON_DIST})")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "correlation_summary.png", dpi=130)
        plt.close(fig)
        made.append("correlation_summary.png")
    return made


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _verdict(rho: float, p: float) -> str:
    if not np.isfinite(rho) or not np.isfinite(p):
        return "insufficient data"
    strength = ("negligible" if abs(rho) < 0.1 else "weak" if abs(rho) < 0.3
                else "moderate" if abs(rho) < 0.5 else "strong")
    sign = "positive" if rho > 0 else "negative"
    sig = "significant" if p < 0.05 else "not significant"
    return f"{strength} {sign} ({sig})"


def _canon_corr(corr: pd.DataFrame, rel: str) -> Dict[str, float]:
    sub = corr[(corr["relationship"] == rel) & (corr["model"] == "ALL")
               & (corr["prototype_type"] == CANON_PTYPE) & (corr["distance"] == CANON_DIST)]
    if len(sub) == 0:
        return {"n": 0, "spearman_rho": np.nan, "spearman_p": np.nan,
                "pearson_r": np.nan, "r2": np.nan}
    r = sub.iloc[0]
    return {"n": int(r["n"]), "spearman_rho": float(r["spearman_rho"]),
            "spearman_p": float(r["spearman_p"]), "pearson_r": float(r["pearson_r"]),
            "r2": float(r["r2"])}


def write_report(metrics: pd.DataFrame, corr: pd.DataFrame, run_status: pd.DataFrame,
                 figures: List[str], report_path: Path, run_id: str) -> None:
    ok = metrics[metrics["status"] == "ok"].copy()
    per_cell = ok.drop_duplicates(subset=["model", "seed", "subject", "direction"])
    n_cells = len(per_cell)
    learned_leak = bool((metrics["used_target_labels_for_training"].astype(str).str.lower()
                         == "true").any()) if len(metrics) else False

    drop_by_model = per_cell.groupby("model")["acc_drop"].agg(["mean", "std", "count"])
    status_counts = run_status["status"].value_counts().to_dict() if len(run_status) else {}

    rel_titles = {
        "prototype_drift_mean_vs_acc_drop": "Prototype drift (mean) vs acc_drop",
        "prototype_direction_cosine_vs_acc_drop": "Prototype direction cosine vs acc_drop",
        "separation_change_vs_acc_drop": "Class separation change vs acc_drop",
        "target_negative_margin_rate_vs_acc_drop": "Target negative-margin rate vs acc_drop",
        "target_margin_mean_vs_acc_drop": "Target margin mean vs acc_drop",
        "fisher_change_vs_acc_drop": "Fisher change vs acc_drop",
    }
    canon = {rel: _canon_corr(corr, rel) for rel in rel_titles}

    L: List[str] = []
    a = L.append
    a("---")
    a('title: "Phase 2c Prototype Drift Analysis Report"')
    a("tags:")
    a('  - "#pipeline/5_dl_model"')
    a('  - "#modality/eeg"')
    a('  - "#method/domain_generalization"')
    a('  - "#paradigm/motor_imagery"')
    a('created: "2026-06-11"')
    a('updated: "2026-06-11"')
    a('status: "active"')
    a("---")
    a("")
    a("# Phase 2c -- Prototype Drift Analysis")
    a("")
    a("> Frozen-model diagnostic: does cross-session accuracy drop come from class-prototype "
      "drift in the penultimate embedding space? (EEGNet / DeepConvNet / FBCNet, WBCIC-SHU 2C)")
    a("")
    a("**target labels are used only for offline diagnostic analysis, not for training or adaptation.**")
    a("")

    a("## 1. Core conclusion (honest)")
    a("")
    dm = canon["prototype_drift_mean_vs_acc_drop"]
    dc = canon["prototype_direction_cosine_vs_acc_drop"]
    nm = canon["target_negative_margin_rate_vs_acc_drop"]
    sc = canon["separation_change_vs_acc_drop"]
    fc = canon["fisher_change_vs_acc_drop"]
    a(f"- Analyzed {n_cells} (subject x direction x model x seed) frozen-model cells "
      f"(canonical subset prototype={CANON_PTYPE}, distance={CANON_DIST}).")
    a(f"- prototype_drift_mean vs acc_drop: Spearman rho={dm['spearman_rho']:.3f} "
      f"(p={dm['spearman_p']:.3g}, n={dm['n']}) -> {_verdict(dm['spearman_rho'], dm['spearman_p'])}.")
    a(f"- prototype_direction_cosine vs acc_drop: rho={dc['spearman_rho']:.3f} "
      f"(p={dc['spearman_p']:.3g}) -> {_verdict(dc['spearman_rho'], dc['spearman_p'])}.")
    a(f"- target_negative_margin_rate vs acc_drop: rho={nm['spearman_rho']:.3f} "
      f"(p={nm['spearman_p']:.3g}) -> {_verdict(nm['spearman_rho'], nm['spearman_p'])}.")
    a(f"- separation_change vs acc_drop: rho={sc['spearman_rho']:.3f} "
      f"(p={sc['spearman_p']:.3g}) -> {_verdict(sc['spearman_rho'], sc['spearman_p'])}.")
    a(f"- fisher_change vs acc_drop: rho={fc['spearman_rho']:.3f} "
      f"(p={fc['spearman_p']:.3g}) -> {_verdict(fc['spearman_rho'], fc['spearman_p'])}.")
    a(f"- Target-label-in-training leakage detected: {learned_leak} (must be False).")
    a("")

    a("## 2. Goal")
    a("")
    a("Test whether the cross-session decode drop is explained by drift of the task "
      "representation (per-class prototypes) in the model's penultimate embedding space, "
      "and which embedding-geometry signal (raw drift, direction consistency, margin, "
      "separation collapse, within-class scatter, Fisher ratio) best predicts the drop. "
      "This is a diagnostic, not an adaptation method.")
    a("")

    a("## 3. Why this experiment follows Phase 2b")
    a("")
    a("Phase 1 found a ~10pp cross-session drop. Phase 2b showed no-learning statistical "
      "alignment (z-score / EA / Riemannian / BN-stats / filterbank) is insufficient (only "
      "BN-stats gave a small positive gain, none reached +2pp, and high-drift subjects were "
      "helped least). That points beyond mean/variance/covariance shift toward task-"
      "representation reorganization -- which this experiment measures directly in embedding space.")
    a("")

    a("## 4. Protocol")
    a("")
    a(f"- Dataset: WBCIC-SHU 2C (left vs right MI), status=ok sessions only, run_id={run_id}.")
    a("- For each subject x directed (source->target) session pair x model x seed: train on "
      "SOURCE session only (train + a stratified val slice from source train for early "
      "stopping); TARGET session is test-only.")
    a("- Subjects with all 3 ok sessions contribute 6 directed pairs; subjects with 2 ok "
      "sessions contribute their available directed pairs; subjects with <2 ok sessions are "
      "skipped (consistent with the Phase 1 cross-session protocol).")
    a("- Seeds: 0,1,2,3,4. Training recipe identical to Phase 1 baseline "
      "(Adam, lr=1e-3, batch=16, max_epochs=100, early-stopping patience=20, val_fraction=0.2).")
    a("- The model is frozen after training; embeddings are extracted with no gradient.")
    a("")

    a("## 5. Model and embedding extraction")
    a("")
    a("- Three baseline architectures share one trainer and one forward contract "
      "`{logits, features, confidence}`.")
    a("- Main embedding = penultimate `features` (flatten before the linear head): "
      "EEGNet/DeepConvNet/FBCNet expose this directly.")
    a("- Auxiliary signals saved per trial: logits, softmax probability, prediction, confidence.")
    a("- None of the three baselines has a learned confidence head, so confidence falls back "
      "to the max softmax probability (documented; not a learned calibration head).")
    a("")

    a("## 6. Prototype definition")
    a("")
    a("Per class, computed separately on SOURCE-train and TARGET-test embeddings:")
    a("- `label_based`: mean(z_i | y_i = class).")
    a("- `confidence_weighted`: confidence-weighted mean of z_i within class.")
    a("- `correct_only`: mean(z_i | y_i = class AND prediction_i = class).")
    a("Distances: euclidean and cosine. (Canonical subset for headline numbers: "
      f"{CANON_PTYPE} / {CANON_DIST}.)")
    a("")

    a("## 7. Leakage control")
    a("")
    a("- Source train/val and target test come from different sessions (asserted).")
    a("- Target labels never enter the training loop, optimizer, or early stopping "
      "(n_target_labels_used_for_training is 0 on every row; used_target_labels_for_training=False).")
    a("- Target X is used only for prediction / embedding extraction; target y is used only "
      "for offline prototype/metric diagnostics.")
    a(f"- Verified across all rows: any target-label-in-training leakage = {learned_leak}.")
    a("")

    a("## 8. Main results")
    a("")
    a("Mean cross-session accuracy drop (source_val - target) per model:")
    a("")
    a("| model | mean acc_drop | std | n_cells |")
    a("|:---|---:|---:|---:|")
    for m, r in drop_by_model.iterrows():
        a(f"| {m} | {r['mean']:.4f} | {r['std']:.4f} | {int(r['count'])} |")
    a("")
    a(f"Total frozen-model cells analyzed: {n_cells}. Run status counts: {status_counts}.")
    a("")

    a("## 9. Correlation analysis")
    a("")
    a("Canonical subset (ALL models, prototype=label_based, distance=euclidean), each vs acc_drop:")
    a("")
    a("| relationship | n | Pearson r | Spearman rho | Spearman p | r2 | verdict |")
    a("|:---|---:|---:|---:|---:|---:|:---|")
    for rel, title in rel_titles.items():
        c = canon[rel]
        a(f"| {title} | {c['n']} | {c['pearson_r']:.3f} | {c['spearman_rho']:.3f} | "
          f"{c['spearman_p']:.3g} | {c['r2']:.3f} | {_verdict(c['spearman_rho'], c['spearman_p'])} |")
    a("")
    a("Full per-(model x prototype_type x distance) breakdown: "
      "`tables/prototype_accuracy_correlation.csv`.")
    a("")

    a("## 10. By-model robustness: EEGNet / DeepConvNet / FBCNet")
    a("")
    a("| model | drift_mean rho | direction_cosine rho | neg_margin rho | separation_change rho | fisher_change rho |")
    a("|:---|---:|---:|---:|---:|---:|")
    for m in sorted(per_cell["model"].unique()):
        def _mrho(metric_rel: str) -> float:
            sub = corr[(corr["relationship"] == metric_rel) & (corr["model"] == m)
                       & (corr["prototype_type"] == CANON_PTYPE) & (corr["distance"] == CANON_DIST)]
            return float(sub.iloc[0]["spearman_rho"]) if len(sub) else float("nan")
        a(f"| {m} | {_mrho('prototype_drift_mean_vs_acc_drop'):.3f} | "
          f"{_mrho('prototype_direction_cosine_vs_acc_drop'):.3f} | "
          f"{_mrho('target_negative_margin_rate_vs_acc_drop'):.3f} | "
          f"{_mrho('separation_change_vs_acc_drop'):.3f} | "
          f"{_mrho('fisher_change_vs_acc_drop'):.3f} |")
    a("")
    a("If the sign/strength differs across models, the prototype-drift story is model-"
      "dependent and must not be over-generalized.")
    a("")

    a("## 11. Interpretation")
    a("")
    a(f"- Does prototype drift explain cross-session drop? raw drift_mean rho="
      f"{dm['spearman_rho']:.3f} ({_verdict(dm['spearman_rho'], dm['spearman_p'])}).")
    a(f"- Is direction cosine more predictive than raw drift distance? "
      f"|rho_dir|={abs(dc['spearman_rho']):.3f} vs |rho_drift|={abs(dm['spearman_rho']):.3f}.")
    a(f"- Does target negative-margin rate explain failures? rho={nm['spearman_rho']:.3f} "
      f"({_verdict(nm['spearman_rho'], nm['spearman_p'])}).")
    a(f"- Does class separation collapse occur / predict drop? separation_change rho="
      f"{sc['spearman_rho']:.3f} ({_verdict(sc['spearman_rho'], sc['spearman_p'])}).")
    a(f"- Does within-class scatter increase / Fisher ratio drop? fisher_change rho="
      f"{fc['spearman_rho']:.3f} ({_verdict(fc['spearman_rho'], fc['spearman_p'])}).")
    a("")

    a("## 12. Relationship to previous phases")
    a("")
    a("- Phase 0: drift is spatial + spectral, not amplitude.")
    a("- Phase 1: ~10pp cross-session drop (EEGNet 0.807->0.711).")
    a("- Phase 2a: multi-source helps but does not close the gap.")
    a("- Phase 2b: statistical alignment insufficient; high-drift subjects helped least.")
    a("- Phase 2c (this): quantifies whether the drop is an embedding-space prototype-drift "
      "phenomenon, which would (or would not) justify prototype-based adaptation.")
    a("")

    a("## 13. Limitations")
    a("")
    a("- Diagnostic only; no adaptation is performed or claimed.")
    a("- Target prototypes use target labels for offline analysis only.")
    a("- Prototypes summarize each class by a single centroid; multi-modal class structure "
      "is not captured.")
    a("- Confidence is fallback (max softmax), not a learned/calibrated head.")
    a("- 2C WBCIC-SHU only; SHU 2022 and cross-dataset are out of scope here.")
    a("")

    a("## 14. Next step")
    a("")
    a("- If prototype drift (and especially direction cosine / negative-margin rate) explains "
      "the drop: prototype-based adaptation (Oracle -> few-shot -> pseudo-label) is justified.")
    a("- If not: investigate decision-boundary drift, within-class scatter growth, class-"
      "separation collapse, or reliability/engagement drift before committing to a prototype "
      "adaptation method.")
    a("")

    a("## 15. File list")
    a("")
    a("- `tables/prototype_drift_metrics.csv` -- main per-cell metric table.")
    a("- `tables/prototype_table.csv` -- per-class prototype metadata (vectors in embedding npz).")
    a("- `tables/prototype_accuracy_correlation.csv` -- correlation analysis.")
    a("- `tables/trial_embeddings_index.csv` -- trial-level embedding index (npz references).")
    a("- `tables/run_status.csv` + `report/RUN_STATUS.md` -- per-cell run status.")
    for f in figures:
        a(f"- `figures/{f}`")
    a("- Heavy artifacts: `outputs/experiments/" + run_id + "/embeddings/` (npz), "
      "`checkpoints/" + run_id + "/`.")
    a("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(L), encoding="utf-8")


def write_run_status_md(run_status: pd.DataFrame, path: Path, run_id: str) -> None:
    counts = run_status["status"].value_counts().to_dict() if len(run_status) else {}
    total = len(run_status)
    L = ["---", 'title: "Phase 2c Run Status"', 'status: "active"', "---", "",
         "# Phase 2c Prototype Drift -- Run Status", "",
         f"run_id: {run_id}", "", f"Total expected cells (subject x direction x model x seed): {total}", "",
         "| status | count |", "|:---|---:|"]
    for k, v in sorted(counts.items()):
        L.append(f"| {k} | {v} |")
    L.append("")
    bad = run_status[run_status["status"].isin(["missing", "failed"])] if total else pd.DataFrame()
    if len(bad) > 0:
        L.append(f"## Missing / failed cells ({len(bad)})")
        L.append("")
        L.append("| model | seed | subject | direction | status |")
        L.append("|:---|---:|:---|:---|:---|")
        for _, r in bad.head(500).iterrows():
            L.append(f"| {r['model']} | {r['seed']} | {r['subject']} | {r['direction']} | {r['status']} |")
        if len(bad) > 500:
            L.append(f"| ... | | | | (+{len(bad) - 500} more) |")
    else:
        L.append("All expected cells present and ok.")
    L.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def summarize(run_dir: Path, readable_dir: Path, *, manifest_path: str,
              models: List[str], seeds: List[int],
              status_filter: Tuple[str, ...] = ("ok",)) -> Dict[str, object]:
    runs_dir = run_dir / "runs"
    tables_dir = readable_dir / "tables"
    figures_dir = readable_dir / "figures"
    report_dir = readable_dir / "report"
    for d in (tables_dir, figures_dir, report_dir):
        d.mkdir(parents=True, exist_ok=True)
    run_id = run_dir.name

    metrics = _concat_glob(runs_dir, "metrics__*.csv")
    prototypes = _concat_glob(runs_dir, "prototypes__*.csv")
    index = _concat_glob(runs_dir, "embed_index__*.csv")
    status = _concat_glob(runs_dir, "status__*.csv")

    # write merged tables (even if empty, so downstream sees the schema)
    metrics.to_csv(tables_dir / "prototype_drift_metrics.csv", index=False)
    prototypes.to_csv(tables_dir / "prototype_table.csv", index=False)
    index.to_csv(tables_dir / "trial_embeddings_index.csv", index=False)

    try:
        expected = _expected_cells(manifest_path, status_filter)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not compute expected cells: {exc}")
        expected = pd.DataFrame(columns=["subject", "source_session", "target_session", "direction"])
    run_status = build_run_status(status, expected, models, seeds)
    run_status.to_csv(tables_dir / "run_status.csv", index=False)
    write_run_status_md(run_status, report_dir / "RUN_STATUS.md", run_id)

    if len(metrics) > 0:
        corr = compute_correlations(metrics)
    else:
        corr = pd.DataFrame()
    corr.to_csv(tables_dir / "prototype_accuracy_correlation.csv", index=False)

    figures = make_figures(metrics, corr, figures_dir) if len(metrics) > 0 else []
    write_report(metrics, corr, run_status,
                 figures, report_dir / "prototype_drift_report.md", run_id)

    summary = {
        "run_id": run_id, "n_metric_rows": int(len(metrics)),
        "n_prototype_rows": int(len(prototypes)), "n_index_rows": int(len(index)),
        "n_status_rows": int(len(status)),
        "run_status_counts": run_status["status"].value_counts().to_dict() if len(run_status) else {},
        "figures": figures, "tables_dir": str(tables_dir), "report_dir": str(report_dir),
    }
    print("[summarize] " + str(summary))
    return summary


def summarize_from_cfg(cfg: Dict) -> Dict[str, object]:
    out_cfg = cfg.get("output", {}) or {}
    run_dir = _abs(out_cfg.get("run_dir", out_cfg.get("output_dir", "outputs/experiments/prototype_drift_v1")))
    readable_dir = _abs(out_cfg.get("readable_dir", "4_experiments/prototype_drift"))
    # Config key is `data.manifest` (logical key or path). Resolved via paths.resolve_manifest_path
    # so SHU never silently falls back to the WBCIC manifest.
    from code.utils.paths import resolve_manifest_path
    manifest_path = str(resolve_manifest_path(cfg))
    models = cfg.get("models", MODELS)
    seeds = [int(s) for s in (cfg.get("seeds") or cfg.get("train", {}).get("seeds", [0]))]
    status_filter = tuple(cfg.get("data", {}).get("status_filter", ["ok"]))
    return summarize(run_dir, readable_dir, manifest_path=manifest_path,
                     models=models, seeds=seeds, status_filter=status_filter)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Phase 2c prototype-drift summarizer")
    ap.add_argument("--run-dir", default="outputs/experiments/prototype_drift_v1")
    ap.add_argument("--readable-dir", default="4_experiments/prototype_drift")
    ap.add_argument("--manifest", default=None, help="processed_manifest.csv (default from paths.yaml)")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--status-filter", default="ok")
    args = ap.parse_args(argv)

    manifest = args.manifest
    if not manifest:
        from code.utils.paths import load_paths
        manifest = str(load_paths(require_raw=False).processed_manifest)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    status_filter = tuple(s.strip() for s in args.status_filter.split(",") if s.strip())
    summarize(_abs(args.run_dir), _abs(args.readable_dir), manifest_path=manifest,
              models=models, seeds=seeds, status_filter=status_filter)


if __name__ == "__main__":
    main()
