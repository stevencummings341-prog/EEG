"""质检与对比可视化：每个函数画一张图并存盘。

输入大多是 pandas DataFrame（每行一个 session 的指标）或聚合好的 numpy 数组，
函数内部不做 IO 之外的重计算。用 Agg 后端，可在无显示的计算节点出图。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_DPI = 130


def _save(fig, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _finite(a: Sequence[float]) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    return arr[np.isfinite(arr)]


# ----------------------------------------------------------------------------
# 1. QC 总览仪表盘
# ----------------------------------------------------------------------------

def plot_qc_dashboard(panels: "Mapping[str, Mapping[str, int]]", out_path: str | Path,
                      title: str = "Preprocessing QC dashboard (eog_ecg_clean)") -> Path:
    """把若干「类别->计数」面板画成柱状图网格（status/label/aux/ICA fallback 等）。"""
    items = list(panels.items())
    n = len(items)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, counts) in zip(axes, items):
        labels = list(counts.keys())
        vals = [counts[k] for k in labels]
        colors = []
        for k in labels:
            kl = str(k).lower()
            if kl in ("ok", "true", "yes", "exact", "pass"):
                colors.append("#2e7d32")
            elif kl in ("failed", "fail", "mismatch"):
                colors.append("#c62828")
            elif kl in ("no_mat", "unknown", "blank"):
                colors.append("#9e9e9e")
            else:
                colors.append("#1565c0")
        bars = ax.bar(range(len(labels)), vals, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_title(name, fontsize=10)
        ax.bar_label(bars, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12, y=1.0)
    fig.tight_layout()
    return _save(fig, out_path)


# ----------------------------------------------------------------------------
# 2. std scatter (ours vs official)
# ----------------------------------------------------------------------------

def plot_std_scatter(df, out_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    x = np.asarray(df.get("official_std"), dtype=np.float64)
    y = np.asarray(df.get("ours_std"), dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    ax.scatter(x, y, s=22, alpha=0.7, edgecolor="k", linewidth=0.3, color="#1565c0")
    if x.size:
        lim = [0, float(np.nanmax([x.max(), y.max()])) * 1.08]
        ax.plot(lim, lim, "--", color="#c62828", lw=1.2, label="y = x")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    ax.set_xlabel("official std (µV)")
    ax.set_ylabel("ours std (µV)")
    ax.set_title("Per-session global std: ours vs official")
    ax.legend()
    ax.grid(alpha=0.25)
    return _save(fig, out_path)


# ----------------------------------------------------------------------------
# 3. std ratio histogram
# ----------------------------------------------------------------------------

def plot_std_ratio_hist(df, out_path: str | Path) -> Path:
    r = _finite(df.get("std_ratio"))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if r.size:
        ax.hist(r, bins=30, color="#1565c0", alpha=0.85, edgecolor="white")
        ax.axvline(1.0, color="#c62828", ls="--", lw=1.3, label="ratio = 1")
        ax.axvline(float(np.median(r)), color="#f9a825", ls="-", lw=1.3,
                   label=f"median = {np.median(r):.3f}")
        ax.legend()
    ax.set_xlabel("ours_std / official_std")
    ax.set_ylabel("# sessions")
    ax.set_title("Distribution of std ratio (ours / official)")
    ax.grid(alpha=0.25)
    return _save(fig, out_path)


# ----------------------------------------------------------------------------
# 4. RMS boxplot ours vs official
# ----------------------------------------------------------------------------

def plot_rms_boxplot(df, out_path: str | Path) -> Path:
    o = _finite(df.get("ours_rms"))
    f = _finite(df.get("official_rms"))
    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot([o, f], labels=["ours", "official"], patch_artist=True,
                    showmeans=True)
    for patch, c in zip(bp["boxes"], ["#1565c0", "#6a1b9a"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel("per-session global RMS (µV)")
    ax.set_title("RMS: ours vs official (per session)")
    ax.grid(alpha=0.25, axis="y")
    return _save(fig, out_path)


# ----------------------------------------------------------------------------
# 5. High-amplitude trial ratio boxplot
# ----------------------------------------------------------------------------

def plot_high_amp_boxplot(df, out_path: str | Path) -> Path:
    thresholds = [100, 200, 500]
    data = []
    labels = []
    for thr in thresholds:
        for side, tag in (("ours", "ours"), ("official", "official")):
            col = f"{side}_high_amp_trial_ratio_{thr}uV"
            data.append(_finite(df.get(col)))
            labels.append(f"{tag}\n>{thr}µV")
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor("#1565c0" if i % 2 == 0 else "#6a1b9a")
        patch.set_alpha(0.55)
    ax.set_ylabel("fraction of trials above threshold")
    ax.set_title("High-amplitude trial ratio: ours vs official")
    ax.grid(alpha=0.25, axis="y")
    return _save(fig, out_path)


# ----------------------------------------------------------------------------
# 6. Global PSD overlay
# ----------------------------------------------------------------------------

def plot_psd_overlay_global(freqs, psd_ours, psd_off, out_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    freqs = np.asarray(freqs)
    ax.semilogy(freqs, psd_ours, color="#1565c0", lw=1.8, label="ours (eog_ecg_clean)")
    if psd_off is not None:
        ax.semilogy(freqs, psd_off, color="#6a1b9a", lw=1.8, label="official derivatives")
    for f0, name in [(10, "mu/alpha"), (20, "beta"), (50, "line")]:
        ax.axvline(f0, color="grey", ls=":", lw=0.8)
    ax.set_xlim(0, min(60, float(freqs.max())))
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD (µV²/Hz)")
    ax.set_title("Global average PSD (mean over channels & sessions)")
    ax.legend()
    ax.grid(alpha=0.25, which="both")
    return _save(fig, out_path)


# ----------------------------------------------------------------------------
# 7. Per-channel PSD overlay (C3/C4/Cz)
# ----------------------------------------------------------------------------

def plot_psd_overlay_channels(freqs, sel_ours: Dict[str, np.ndarray],
                              sel_off: Dict[str, np.ndarray], out_path: str | Path) -> Path:
    chans = [c for c in ("C3", "C4", "Cz") if c in sel_ours]
    if not chans:
        chans = list(sel_ours.keys())
    n = max(1, len(chans))
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.4), squeeze=False)
    freqs = np.asarray(freqs)
    for ax, ch in zip(axes[0], chans):
        ax.semilogy(freqs, sel_ours[ch], color="#1565c0", lw=1.7, label="ours")
        if sel_off and ch in sel_off:
            ax.semilogy(freqs, sel_off[ch], color="#6a1b9a", lw=1.7, label="official")
        ax.axvspan(8, 13, color="#fff59d", alpha=0.3)
        ax.axvspan(13, 30, color="#c8e6c9", alpha=0.3)
        ax.set_xlim(0, min(60, float(freqs.max())))
        ax.set_xlabel("freq (Hz)")
        ax.set_title(f"PSD @ {ch}")
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("PSD (µV²/Hz)")
    fig.suptitle("Per-channel average PSD (mu shaded yellow, beta green)", y=1.02)
    fig.tight_layout()
    return _save(fig, out_path)


# ----------------------------------------------------------------------------
# 8. Bandpower ratio mu / beta
# ----------------------------------------------------------------------------

def plot_bandpower_ratio_mu_beta(df, out_path: str | Path) -> Path:
    mu = _finite(df.get("mu_bandpower_ratio"))
    beta = _finite(df.get("beta_bandpower_ratio"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, vals, name in ((axes[0], mu, "mu/alpha (8–13 Hz)"),
                           (axes[1], beta, "beta (13–30 Hz)")):
        if vals.size:
            ax.hist(vals, bins=25, color="#1565c0", alpha=0.85, edgecolor="white")
            ax.axvline(1.0, color="#c62828", ls="--", lw=1.3, label="ratio = 1")
            ax.axvline(float(np.median(vals)), color="#f9a825", lw=1.3,
                       label=f"median = {np.median(vals):.3f}")
            ax.legend(fontsize=8)
        ax.set_title(f"bandpower ratio ours/official: {name}")
        ax.set_xlabel("ratio")
        ax.set_ylabel("# sessions")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    return _save(fig, out_path)


# ----------------------------------------------------------------------------
# 9. Per-channel RMS ratio heatmap
# ----------------------------------------------------------------------------

def plot_channel_rms_ratio_heatmap(ratio_matrix: np.ndarray, session_labels: List[str],
                                   channel_names: List[str], out_path: str | Path) -> Path:
    M = np.asarray(ratio_matrix, dtype=np.float64)  # [sessions, channels]
    fig, ax = plt.subplots(figsize=(max(10, len(channel_names) * 0.22),
                                    max(6, len(session_labels) * 0.10)))
    vmax = 2.0
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=0.0, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(range(len(channel_names)))
    ax.set_xticklabels(channel_names, rotation=90, fontsize=5)
    # session ticks 太多，稀疏标注
    step = max(1, len(session_labels) // 40)
    ax.set_yticks(range(0, len(session_labels), step))
    ax.set_yticklabels([session_labels[i] for i in range(0, len(session_labels), step)],
                       fontsize=5)
    ax.set_xlabel("channel")
    ax.set_ylabel("session")
    ax.set_title("Per-channel RMS ratio (ours / official); 1.0 = match, <1 = ours lower")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("RMS ratio")
    return _save(fig, out_path)


# ----------------------------------------------------------------------------
# 10. Trial correlation histogram (exact-label sessions)
# ----------------------------------------------------------------------------

def plot_trial_corr_hist(mean_trial_corr: Sequence[float], out_path: str | Path,
                         n_exact: Optional[int] = None) -> Path:
    vals = _finite(mean_trial_corr)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if vals.size:
        ax.hist(vals, bins=25, color="#2e7d32", alpha=0.85, edgecolor="white")
        ax.axvline(float(np.median(vals)), color="#f9a825", lw=1.4,
                   label=f"median = {np.median(vals):.3f}")
        ax.legend()
    sub = f" ({n_exact} exact-label sessions)" if n_exact is not None else ""
    ax.set_title(f"Mean trial-wise correlation: ours vs official{sub}")
    ax.set_xlabel("mean trial Pearson r (per session)")
    ax.set_ylabel("# sessions")
    ax.grid(alpha=0.25)
    return _save(fig, out_path)


# ----------------------------------------------------------------------------
# 11. Example waveform overlay
# ----------------------------------------------------------------------------

def plot_example_waveforms(examples: List[Dict[str, Any]], out_path: str | Path) -> Path:
    n = max(1, len(examples))
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.4 * n), squeeze=False)
    for ax, ex in zip(axes[:, 0], examples):
        t = ex["t"]
        ax.plot(t, ex["ours"], color="#1565c0", lw=1.0, label="ours")
        ax.plot(t, ex["official"], color="#6a1b9a", lw=1.0, alpha=0.8, label="official")
        ax.set_title(ex.get("title", ""), fontsize=9)
        ax.set_ylabel("µV")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("time (s)")
    fig.suptitle("Example waveform overlay (ours vs official)", y=1.005)
    fig.tight_layout()
    return _save(fig, out_path)


# ----------------------------------------------------------------------------
# 12. Class mu/beta difference at C3/C4 (separability comparison)
# ----------------------------------------------------------------------------

def plot_class_mu_beta_difference(df, out_path: str | Path) -> Path:
    """比较 ours 与 official 在 C3/C4 上 mu/beta 的两类可分性（|Cohen's d|）。"""
    pairs = [("C3", "mu_alpha"), ("C3", "beta"), ("C4", "mu_alpha"), ("C4", "beta")]
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.6 * len(pairs), 4.6), squeeze=False)
    for ax, (ch, band) in zip(axes[0], pairs):
        oc = f"ours_{ch}_{band}_cohend"
        fc = f"official_{ch}_{band}_cohend"
        x = np.abs(_pair(df, fc))
        y = np.abs(_pair(df, oc))
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        ax.scatter(x, y, s=20, alpha=0.7, color="#00695c", edgecolor="k", linewidth=0.3)
        if x.size:
            lim = [0, float(np.nanmax([x.max(), y.max(), 0.1])) * 1.1]
            ax.plot(lim, lim, "--", color="#c62828", lw=1.0)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
        ax.set_xlabel(f"official |d|")
        ax.set_ylabel(f"ours |d|")
        band_disp = "mu" if band == "mu_alpha" else "beta"
        ax.set_title(f"{ch} {band_disp} class |Cohen's d|", fontsize=10)
        ax.grid(alpha=0.25)
    fig.suptitle("MI class separability (|Cohen's d| on log-bandpower): ours vs official", y=1.02)
    fig.tight_layout()
    return _save(fig, out_path)


def _pair(df, col) -> np.ndarray:
    if col in getattr(df, "columns", []):
        return np.asarray(df[col], dtype=np.float64)
    return np.full(len(df), np.nan)


# ----------------------------------------------------------------------------
# 13. ICA excluded components summary
# ----------------------------------------------------------------------------

def plot_ica_excluded_summary(eog_counts: Sequence[int], ecg_counts: Sequence[int],
                              total_counts: Sequence[int], out_path: str | Path) -> Path:
    eog = np.asarray(eog_counts, dtype=int)
    ecg = np.asarray(ecg_counts, dtype=int)
    tot = np.asarray(total_counts, dtype=int)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    specs = [(axes[0], eog, "EOG excluded ICs / session", "#1565c0"),
             (axes[1], ecg, "ECG excluded ICs / session", "#6a1b9a"),
             (axes[2], tot, "Total excluded ICs / session", "#2e7d32")]
    for ax, arr, title, color in specs:
        if arr.size:
            maxv = int(arr.max())
            bins = np.arange(-0.5, maxv + 1.5, 1.0)
            ax.hist(arr, bins=bins, color=color, alpha=0.85, edgecolor="white")
            ax.set_xticks(range(0, maxv + 1))
            ax.axvline(float(arr.mean()), color="#f9a825", lw=1.3,
                       label=f"mean = {arr.mean():.2f}")
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("# components")
        ax.set_ylabel("# sessions")
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("ICA excluded components (EOG / ECG / total)", y=1.02)
    fig.tight_layout()
    return _save(fig, out_path)
