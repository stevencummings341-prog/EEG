"""Cross-session distribution-drift diagnostics (direction A).

Quantifies *what* changes across sessions of the same subject, to explain why
cross-session MI generalization is hard. Re-implemented from the senior's
``session_drift_diagnostic.py`` (docs/references/senior_scripts/...) with these
project-grade changes:

  * status=ok filtering via processed_manifest.csv (not a hard-coded failed-session
    set); only ok sessions are ever loaded.
  * labels normalized to {0,1} (the reference assumed {1,2}); metrics that use labels
    (Fisher ratio, CSP) work with either storage.
  * matplotlib-only (no seaborn hard dependency).
  * vectorized FFT band-power / ERD-ERS (much faster than the per-trial Python loops).
  * numpy/scipy/sklearn only — a CPU-only job (no torch, no GPU).

Data shapes / units:
  * Each ok session .npz: X [n_trials, 58, 1000] float32 (µV @ 250 Hz), y [n_trials].
  * Metrics are computed for every WITHIN-subject session pair (i<j, undirected).

Metrics (see docs/SESSION_DRIFT_ANALYSIS.md for definitions):
  mmd, coral, mu_power_shift, beta_power_shift, mu_ks_stat, erd_mu_corr, erd_beta_corr,
  csp_similarity, rms_ratio_median, rms_ratio_std, fisher_i, fisher_j, fisher_shift,
  plus signal-quality (high-amplitude trial ratio, mean RMS) per session.

Outputs (written by scripts/analysis/run_session_drift.py):
  session_drift_report.csv, summary.json, SESSION_DRIFT_REPORT.md, figures/*.png
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # headless: no display on compute nodes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist

from ..data.session_splits import SessionRecord, group_by_subject, normalize_labels
from ..utils.io import load_session_npz


@dataclass
class DriftParams:
    """Drift-analysis parameters (from configs/session_drift.yaml)."""

    fs: int = 250
    mu_band: Tuple[float, float] = (8.0, 13.0)
    beta_band: Tuple[float, float] = (13.0, 30.0)
    mmd_subsample: int = 100
    csp_components: int = 4
    erd_baseline_ratio: float = 0.25
    seed: int = 0
    high_amp_uv: float = 100.0       # |amplitude| threshold for "high-amp" trials (µV)


# --------------------------------------------------------------------------- #
# Spectral features (vectorized)
# --------------------------------------------------------------------------- #
def band_log_power(X: np.ndarray, fs: int, band: Tuple[float, float]) -> np.ndarray:
    """Per-trial, per-channel log10 band power. X [trials, ch, time] -> [trials, ch]."""
    n_times = X.shape[-1]
    freqs = np.fft.rfftfreq(n_times, d=1.0 / fs)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    F = np.fft.rfft(X, axis=-1)
    psd = (F.real ** 2 + F.imag ** 2)
    bp = psd[..., mask].mean(axis=-1)
    return np.log10(bp + 1e-10)


def erd_ers_pattern(X: np.ndarray, fs: int, band: Tuple[float, float],
                    baseline_ratio: float) -> np.ndarray:
    """Median ERD/ERS (%) per channel. Baseline = first ``baseline_ratio`` of the trial.

    Returns [channels]: (mi_power - baseline_power) / baseline_power * 100, median over trials.
    Negative => desynchronization (ERD), positive => synchronization (ERS).
    """
    n_times = X.shape[-1]
    bl = max(1, int(n_times * baseline_ratio))

    def seg_band_power(seg: np.ndarray) -> np.ndarray:
        nf = seg.shape[-1]
        freqs = np.fft.rfftfreq(nf, d=1.0 / fs)
        mask = (freqs >= band[0]) & (freqs <= band[1])
        F = np.fft.rfft(seg, axis=-1)
        psd = (F.real ** 2 + F.imag ** 2)
        return psd[..., mask].mean(axis=-1) + 1e-10   # [trials, ch]

    blp = seg_band_power(X[..., :bl])
    mip = seg_band_power(X[..., bl:])
    erd = (mip - blp) / blp * 100.0
    return np.median(erd, axis=0)


# --------------------------------------------------------------------------- #
# Distribution distances
# --------------------------------------------------------------------------- #
def mmd_rbf(X1: np.ndarray, X2: np.ndarray, subsample: int, rng: np.random.Generator) -> float:
    """RBF-kernel MMD between two trial sets (flattened), median-heuristic gamma."""
    f1 = X1.reshape(X1.shape[0], -1)
    f2 = X2.reshape(X2.shape[0], -1)
    n = min(subsample, f1.shape[0], f2.shape[0])
    if n < 2:
        return float("nan")
    i1 = rng.choice(f1.shape[0], n, replace=False)
    i2 = rng.choice(f2.shape[0], n, replace=False)
    f1, f2 = f1[i1], f2[i2]
    D = cdist(f1, f2, "sqeuclidean")
    med = np.median(D[D > 0]) if np.any(D > 0) else 1.0
    gamma = 1.0 / med if med > 0 else 1.0
    k11 = np.exp(-gamma * cdist(f1, f1, "sqeuclidean"))
    k22 = np.exp(-gamma * cdist(f2, f2, "sqeuclidean"))
    k12 = np.exp(-gamma * cdist(f1, f2, "sqeuclidean"))
    mmd_sq = k11.mean() + k22.mean() - 2.0 * k12.mean()
    return float(np.sqrt(max(mmd_sq, 0.0)))


def coral_distance(X1: np.ndarray, X2: np.ndarray) -> float:
    """CORAL: Frobenius distance between per-trial channel-mean covariances, / n_ch."""
    s1 = X1.mean(axis=2)   # [trials, ch]
    s2 = X2.mean(axis=2)
    c1 = np.cov(s1.T) + 1e-6 * np.eye(s1.shape[1])
    c2 = np.cov(s2.T) + 1e-6 * np.eye(s2.shape[1])
    diff = c1 - c2
    return float(np.sqrt(np.sum(diff ** 2)) / diff.shape[0])


def fisher_ratio(X: np.ndarray, y: np.ndarray, fs: int, band: Tuple[float, float]) -> float:
    """Mean per-channel Fisher ratio (between/within class variance) on mu log-power."""
    feat = band_log_power(X, fs, band)   # [trials, ch]
    y = normalize_labels(y)
    classes = np.unique(y)
    if len(classes) < 2:
        return 0.0
    ratios = []
    overall = feat.mean(axis=0)          # [ch]
    for ch in range(feat.shape[1]):
        f = feat[:, ch]
        between = within = 0.0
        for c in classes:
            fc = f[y == c]
            nc = fc.size
            between += nc * (fc.mean() - overall[ch]) ** 2
            within += nc * fc.var()
        ratios.append(between / (within + 1e-10))
    return float(np.mean(ratios))


# --------------------------------------------------------------------------- #
# Spatial patterns
# --------------------------------------------------------------------------- #
def _fit_csp_filters(X: np.ndarray, y: np.ndarray, n_components: int) -> Optional[np.ndarray]:
    """Simple 2-class CSP filters [channels, n_components] (no mne dependency)."""
    y = normalize_labels(y)
    classes = np.unique(y)
    if len(classes) != 2:
        return None
    covs = []
    for c in classes:
        Xc = X[y == c]                       # [n_c, ch, time]
        flat = Xc.transpose(1, 0, 2).reshape(Xc.shape[1], -1)  # [ch, n_c*time]
        cov = np.cov(flat) + 1e-6 * np.eye(Xc.shape[1])
        covs.append(cov)
    Csum = covs[0] + covs[1]
    try:
        eigvals, eigvecs = np.linalg.eigh(np.linalg.inv(Csum) @ covs[0])
    except np.linalg.LinAlgError:
        return None
    order = np.argsort(eigvals)[::-1]
    half = max(1, n_components // 2)
    sel = np.concatenate([order[:half], order[-half:]])
    return eigvecs[:, sel]


def csp_similarity(X1, y1, X2, y2, n_components: int) -> float:
    """Mean of the top-|cosine| similarities between two sessions' CSP filter sets."""
    f1 = _fit_csp_filters(X1, y1, n_components)
    f2 = _fit_csp_filters(X2, y2, n_components)
    if f1 is None or f2 is None:
        return float("nan")
    sims = []
    for i in range(f1.shape[1]):
        for j in range(f2.shape[1]):
            num = abs(float(np.dot(f1[:, i], f2[:, j])))
            den = np.linalg.norm(f1[:, i]) * np.linalg.norm(f2[:, j]) + 1e-10
            sims.append(num / den)
    sims = sorted(sims, reverse=True)
    k = min(n_components, len(sims))
    return float(np.mean(sims[:k]))


def channel_rms_ratio(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """Per-channel RMS ratio RMS(X2)/RMS(X1). Returns [channels]."""
    rms1 = np.sqrt(np.mean(X1 ** 2, axis=(0, 2)))
    rms2 = np.sqrt(np.mean(X2 ** 2, axis=(0, 2)))
    return rms2 / (rms1 + 1e-10)


def signal_quality(X: np.ndarray, high_amp_uv: float) -> Dict[str, float]:
    """High-amplitude trial ratio + mean/std RMS for one session."""
    rms = np.sqrt(np.mean(X ** 2, axis=(1, 2)))
    high = np.max(np.abs(X), axis=(1, 2))
    return {
        "high_amp_ratio": float(np.mean(high > high_amp_uv)),
        "mean_rms": float(np.mean(rms)),
        "std_rms": float(np.std(rms)),
    }


# --------------------------------------------------------------------------- #
# Per-pair metric computation
# --------------------------------------------------------------------------- #
def compute_pair_metrics(Xi, yi, Xj, yj, params: DriftParams,
                         rng: np.random.Generator) -> Dict[str, float]:
    """All drift metrics for one (session_i, session_j) pair of the same subject."""
    row: Dict[str, float] = {}
    row["mmd"] = mmd_rbf(Xi, Xj, params.mmd_subsample, rng)
    row["coral"] = coral_distance(Xi, Xj)

    mu_i = band_log_power(Xi, params.fs, params.mu_band)
    mu_j = band_log_power(Xj, params.fs, params.mu_band)
    be_i = band_log_power(Xi, params.fs, params.beta_band)
    be_j = band_log_power(Xj, params.fs, params.beta_band)
    row["mu_power_shift"] = float(mu_j.mean() - mu_i.mean())
    row["beta_power_shift"] = float(be_j.mean() - be_i.mean())
    row["mu_ks_stat"] = float(stats.ks_2samp(mu_i.ravel(), mu_j.ravel()).statistic)

    erd_mu_i = erd_ers_pattern(Xi, params.fs, params.mu_band, params.erd_baseline_ratio)
    erd_mu_j = erd_ers_pattern(Xj, params.fs, params.mu_band, params.erd_baseline_ratio)
    row["erd_mu_corr"] = float(np.corrcoef(erd_mu_i, erd_mu_j)[0, 1])
    erd_be_i = erd_ers_pattern(Xi, params.fs, params.beta_band, params.erd_baseline_ratio)
    erd_be_j = erd_ers_pattern(Xj, params.fs, params.beta_band, params.erd_baseline_ratio)
    row["erd_beta_corr"] = float(np.corrcoef(erd_be_i, erd_be_j)[0, 1])

    row["csp_similarity"] = csp_similarity(Xi, yi, Xj, yj, params.csp_components)

    rms_ratio = channel_rms_ratio(Xi, Xj)
    row["rms_ratio_median"] = float(np.median(rms_ratio))
    row["rms_ratio_std"] = float(np.std(rms_ratio))

    row["fisher_i"] = fisher_ratio(Xi, yi, params.fs, params.mu_band)
    row["fisher_j"] = fisher_ratio(Xj, yj, params.fs, params.mu_band)
    row["fisher_shift"] = float(row["fisher_j"] - row["fisher_i"])

    qi = signal_quality(Xi, params.high_amp_uv)
    qj = signal_quality(Xj, params.high_amp_uv)
    row["high_amp_ratio_i"] = qi["high_amp_ratio"]
    row["high_amp_ratio_j"] = qj["high_amp_ratio"]
    row["mean_rms_i"] = qi["mean_rms"]
    row["mean_rms_j"] = qj["mean_rms"]
    return row


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_drift_analysis(
    records: Sequence[SessionRecord],
    params: DriftParams,
    logger=None,
) -> pd.DataFrame:
    """Compute drift metrics for every within-subject session pair. Returns a DataFrame."""
    by_subj = group_by_subject(records)
    rng = np.random.default_rng(params.seed)
    rows: List[Dict[str, object]] = []
    cache: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}

    def _load(rec: SessionRecord):
        key = rec.key
        if key not in cache:
            d = load_session_npz(rec.npz_path)
            cache[key] = (np.asarray(d["X"], dtype=np.float64), np.asarray(d["y"]))
        return cache[key]

    n_pairs = 0
    for subj, recs in by_subj.items():
        if len(recs) < 2:
            if logger:
                logger.info("%s: <2 ok sessions; no pairs.", subj)
            continue
        for ri, rj in itertools.combinations(recs, 2):
            Xi, yi = _load(ri)
            Xj, yj = _load(rj)
            m = compute_pair_metrics(Xi, yi, Xj, yj, params, rng)
            m.update({"subject": subj, "ses_i": ri.session, "ses_j": rj.session})
            rows.append(m)
            n_pairs += 1
            if logger and n_pairs % 10 == 0:
                logger.info("processed %d session pairs...", n_pairs)
        # free this subject's cache to bound memory
        for r in recs:
            cache.pop(r.key, None)

    if not rows:
        raise ValueError("No session pairs computed (need subjects with >=2 ok sessions).")
    df = pd.DataFrame(rows)
    front = ["subject", "ses_i", "ses_j"]
    df = df[front + [c for c in df.columns if c not in front]]
    if logger:
        logger.info("drift metrics computed for %d pairs / %d subjects.",
                    len(df), df["subject"].nunique())
    return df


# --------------------------------------------------------------------------- #
# Summary + report
# --------------------------------------------------------------------------- #
_SUMMARY_METRICS = [
    "mmd", "coral", "mu_power_shift", "beta_power_shift", "mu_ks_stat",
    "erd_mu_corr", "erd_beta_corr", "csp_similarity",
    "rms_ratio_median", "fisher_i", "fisher_j", "fisher_shift",
]


def summarize(df: pd.DataFrame, params: DriftParams) -> Dict[str, object]:
    """Mean/median/std per metric + counts (for summary.json)."""
    stats_out: Dict[str, Dict[str, float]] = {}
    for m in _SUMMARY_METRICS:
        if m in df.columns:
            s = df[m].dropna()
            stats_out[m] = {
                "mean": float(s.mean()) if len(s) else float("nan"),
                "median": float(s.median()) if len(s) else float("nan"),
                "std": float(s.std()) if len(s) else float("nan"),
                "n": int(len(s)),
            }
    return {
        "n_pairs": int(len(df)),
        "n_subjects": int(df["subject"].nunique()),
        "bands": {"mu": list(params.mu_band), "beta": list(params.beta_band)},
        "params": {
            "fs": params.fs, "mmd_subsample": params.mmd_subsample,
            "csp_components": params.csp_components,
            "erd_baseline_ratio": params.erd_baseline_ratio, "seed": params.seed,
        },
        "metrics": stats_out,
    }


def _hist(ax, data, title, xlabel, vline=None, vline_label=None):
    data = np.asarray(pd.Series(data).dropna())
    if data.size:
        ax.hist(data, bins=min(30, max(5, data.size // 2)), edgecolor="black", alpha=0.75)
    if vline is not None:
        ax.axvline(vline, color="red", linestyle="--", label=vline_label)
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("session pairs")


def generate_figures(df: pd.DataFrame, fig_dir: Path, logger=None) -> List[str]:
    """Write drift figures (matplotlib-only). Returns the list of filenames written."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    def _save(fig, name):
        fig.tight_layout()
        fig.savefig(fig_dir / name, dpi=150)
        plt.close(fig)
        written.append(name)
        if logger:
            logger.info("  figure: %s", name)

    # 1. distribution distances
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    _hist(axes[0], df.get("mmd"), "Cross-session MMD", "MMD")
    _hist(axes[1], df.get("coral"), "Cross-session CORAL", "CORAL distance")
    _save(fig, "distribution_distance_hist.png")

    # 2. band-power shift
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    _hist(axes[0], df.get("mu_power_shift"), "mu power shift", "log10 power diff (j-i)", 0.0, "no shift")
    _hist(axes[1], df.get("beta_power_shift"), "beta power shift", "log10 power diff (j-i)", 0.0, "no shift")
    _save(fig, "band_power_shift_hist.png")

    # 3. ERD/ERS spatial-pattern correlation
    fig, ax = plt.subplots(figsize=(8, 5))
    for col, lab in [("erd_mu_corr", "mu"), ("erd_beta_corr", "beta")]:
        d = np.asarray(df.get(col, pd.Series(dtype=float)).dropna())
        if d.size:
            ax.hist(d, bins=20, alpha=0.55, edgecolor="black", label=f"{lab} ERD/ERS")
    ax.axvline(1.0, color="red", linestyle="--", label="perfect")
    ax.set_title("Cross-session ERD/ERS spatial-pattern correlation")
    ax.set_xlabel("Pearson corr between sessions"); ax.set_ylabel("session pairs"); ax.legend()
    _save(fig, "erd_ers_correlation_hist.png")

    # 4. CSP similarity
    fig, ax = plt.subplots(figsize=(8, 5))
    _hist(ax, df.get("csp_similarity"), "Cross-session CSP filter similarity", "|cosine| similarity",
          1.0, "perfect")
    _save(fig, "csp_similarity_hist.png")

    # 5. Fisher separability scatter
    if {"fisher_i", "fisher_j"} <= set(df.columns):
        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.scatter(df["fisher_i"], df["fisher_j"], alpha=0.5, s=20)
        lim = float(np.nanmax([df["fisher_i"].max(), df["fisher_j"].max()])) * 1.1 + 1e-6
        ax.plot([0, lim], [0, lim], "r--", label="y=x (no change)")
        ax.set_xlabel("Fisher ratio (session i)"); ax.set_ylabel("Fisher ratio (session j)")
        ax.set_title("Cross-session MI separability"); ax.legend()
        _save(fig, "fisher_ratio_scatter.png")

    # 6. RMS ratio
    fig, ax = plt.subplots(figsize=(8, 5))
    _hist(ax, df.get("rms_ratio_median"), "Cross-session amplitude (RMS) ratio",
          "median channel RMS ratio (j/i)", 1.0, "no change")
    _save(fig, "rms_ratio_hist.png")

    # 7. metric correlation matrix
    cols = [c for c in ["mmd", "coral", "mu_power_shift", "beta_power_shift", "mu_ks_stat",
                        "erd_mu_corr", "csp_similarity", "fisher_shift", "rms_ratio_median"]
            if c in df.columns and df[c].notna().sum() > 5]
    if len(cols) >= 3:
        corr = df[cols].corr().values
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=8)
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title("Drift-metric correlation matrix")
        _save(fig, "metric_correlation_matrix.png")

    # 8. session-pair-type comparison (e.g. 1-2, 1-3, 2-3)
    d = df.copy()
    try:
        d["pair_type"] = (d["ses_i"].str.extract(r"(\d+)").astype(int).iloc[:, 0].astype(str)
                          + "-" + d["ses_j"].str.extract(r"(\d+)").astype(int).iloc[:, 0].astype(str))
        order = sorted(d["pair_type"].unique())
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, col, lab in zip(axes, ["mmd", "fisher_shift"], ["MMD", "Fisher shift"]):
            groups = [d.loc[d["pair_type"] == pt, col].dropna().values for pt in order]
            ax.boxplot(groups, labels=order)
            ax.set_xlabel("session pair"); ax.set_ylabel(lab); ax.set_title(f"{lab} by session pair")
        _save(fig, "session_pair_comparison.png")
    except Exception as e:  # noqa: BLE001 - plotting must never crash the whole run
        if logger:
            logger.warning("session_pair_comparison figure skipped: %s", e)

    return written


def write_markdown_report(summary: Dict[str, object], figures: List[str], out_path: Path,
                          report_run_id: str) -> None:
    """Write a human-readable SESSION_DRIFT_REPORT.md from the summary + figure list."""
    m = summary["metrics"]

    def cell(metric, key):
        return f"{m[metric][key]:.4f}" if metric in m else "—"

    lines: List[str] = []
    lines.append(f"# Session Drift Report — {report_run_id}\n")
    lines.append(f"- Session pairs analyzed: **{summary['n_pairs']}** across "
                 f"**{summary['n_subjects']}** subjects (status=ok only).")
    lines.append(f"- Bands: mu = {summary['bands']['mu']} Hz, beta = {summary['bands']['beta']} Hz; "
                 f"fs = {summary['params']['fs']} Hz.\n")
    lines.append("## Metric summary (mean / median / std)\n")
    lines.append("| metric | mean | median | std | what it means |")
    lines.append("|---|---|---|---|---|")
    meanings = {
        "mmd": "overall distribution distance (higher = more drift)",
        "coral": "channel-covariance distance (higher = more drift)",
        "mu_power_shift": "mu power change j-i (0 = stable)",
        "beta_power_shift": "beta power change j-i (0 = stable)",
        "mu_ks_stat": "mu-power distribution shift (0 = identical)",
        "erd_mu_corr": "mu ERD/ERS spatial stability (1 = identical pattern)",
        "erd_beta_corr": "beta ERD/ERS spatial stability (1 = identical)",
        "csp_similarity": "spatial-filter stability (1 = identical)",
        "rms_ratio_median": "amplitude ratio j/i (1 = no change)",
        "fisher_i": "MI separability in session i",
        "fisher_j": "MI separability in session j",
        "fisher_shift": "separability change j-i (sign = direction)",
    }
    for met in _SUMMARY_METRICS:
        if met in m:
            lines.append(f"| `{met}` | {cell(met,'mean')} | {cell(met,'median')} | "
                         f"{cell(met,'std')} | {meanings.get(met,'')} |")
    lines.append("\n## How to read it\n")
    lines.append("- **High MMD/CORAL, low CSP/ERD-ERS correlation** ⇒ spatial pattern drifts "
                 "→ favor alignment (Euclidean Alignment / CORAL).")
    lines.append("- **Large `rms_ratio` deviation from 1 / mu-beta power shift** ⇒ amplitude/"
                 "spectral drift → favor BN adaptation / re-normalization / filter-bank alignment.")
    lines.append("- **Positive `fisher_shift` on later sessions** ⇒ MI separability improves "
                 "(learning effect) → favor online/test-time adaptation.")
    lines.append("- These point to which adaptation mechanism the cross-session model should use.\n")
    lines.append("## Figures\n")
    for f in figures:
        lines.append(f"- `figures/{f}`")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
