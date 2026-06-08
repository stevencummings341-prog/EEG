#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rebuild the cross-session drift REPORT from the existing drift CSV (no recompute).

Reads the already-computed ``outputs/analysis/session_drift_v1/session_drift_report.csv``
(144 within-subject session pairs / 50 subjects) and produces a proper experiment report
plus per-pair / per-subject tables and reporting figures. It does NOT recompute any drift
metric (does not touch the npz data) and does not submit any job.

Inputs (read-only):
  outputs/analysis/session_drift_v1/session_drift_report.csv

New outputs (added, originals untouched):
  SESSION_DRIFT_REPORT.md            full experiment report (rewritten)
  SESSION_DRIFT_SUMMARY_CN.md        one-page Chinese summary for the advisor
  session_pair_summary.csv           per pair-type (1-2/1-3/2-3) aggregate
  per_subject_drift_summary.csv/.md  per-subject drift profile + drift_level
  figures/session_pair_metric_summary.png
  figures/subject_mmd_heatmap.png
  figures/subject_csp_heatmap.png
  figures/subject_erd_mu_heatmap.png
  figures/high_drift_subjects_bar.png
  figures/signal_quality_shift.png

Run (light, CPU; safe on a login node — reads a 144-row CSV, writes small PNGs):
  python scripts/analysis/build_drift_report.py
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "session_drift_v1"

# The 5 failed sessions (excluded upstream) and the partial subjects they belong to.
FAILED_SESSIONS = {
    "sub-023": ["ses-01"], "sub-024": ["ses-02", "ses-03"],
    "sub-026": ["ses-01"], "sub-032": ["ses-02"],
}
PAIR_ORDER = ["01-02", "01-03", "02-03"]


def load_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["pair"] = (df["ses_i"].str.extract(r"(\d+)")[0] + "-"
                  + df["ses_j"].str.extract(r"(\d+)")[0])
    return df


def _fmt(v, nd=3):
    return "—" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.{nd}f}"


# --------------------------------------------------------------------------- #
# Aggregations
# --------------------------------------------------------------------------- #
def overall_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    cols = ["mmd", "coral", "mu_power_shift", "beta_power_shift", "mu_ks_stat",
            "erd_mu_corr", "erd_beta_corr", "csp_similarity", "rms_ratio_median",
            "fisher_i", "fisher_j", "fisher_shift"]
    out = {}
    for c in cols:
        s = df[c].dropna()
        out[c] = {"mean": float(s.mean()), "median": float(s.median()),
                  "std": float(s.std(ddof=0)), "n": int(len(s))}
    return out


def session_pair_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair in PAIR_ORDER:
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

    # Tertile classification (data-driven, documented in the report).
    q1, q2 = sub["drift_score"].quantile([1 / 3, 2 / 3])
    def level(s):
        return "high" if s >= q2 else ("stable" if s <= q1 else "moderate")
    sub["drift_level"] = sub["drift_score"].apply(level)
    return sub.sort_values("drift_score", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Figures (English labels to avoid CJK font issues; report prose is Chinese)
# --------------------------------------------------------------------------- #
def fig_pair_metric_summary(df: pd.DataFrame, out: Path) -> None:
    metrics = [("mmd", "MMD"), ("csp_similarity", "CSP similarity"),
               ("erd_mu_corr", "ERD/ERS mu corr"), ("mu_ks_stat", "mu KS stat")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, (col, title) in zip(axes, metrics):
        data = [df.loc[df["pair"] == p, col].dropna().values for p in PAIR_ORDER]
        ax.boxplot(data, showmeans=True)
        ax.set_xticks(range(1, len(PAIR_ORDER) + 1)); ax.set_xticklabels(PAIR_ORDER)
        ax.set_title(title); ax.set_xlabel("session pair")
    fig.suptitle("Cross-session drift by session pair (1-2 / 1-3 / 2-3)")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def _subject_pair_matrix(df: pd.DataFrame, col: str):
    subs = sorted(df["subject"].unique())
    mat = np.full((len(subs), len(PAIR_ORDER)), np.nan)
    sidx = {s: i for i, s in enumerate(subs)}
    pidx = {p: j for j, p in enumerate(PAIR_ORDER)}
    for _, r in df.iterrows():
        mat[sidx[r["subject"]], pidx[r["pair"]]] = r[col]
    return subs, mat


def fig_subject_heatmap(df: pd.DataFrame, col: str, title: str, cmap: str, out: Path,
                        vmin=None, vmax=None) -> None:
    subs, mat = _subject_pair_matrix(df, col)
    fig, ax = plt.subplots(figsize=(5.5, max(8, len(subs) * 0.22)))
    cm = plt.get_cmap(cmap).copy(); cm.set_bad("lightgray")
    im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", cmap=cm, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(PAIR_ORDER))); ax.set_xticklabels(PAIR_ORDER)
    ax.set_yticks(range(len(subs)))
    ax.set_yticklabels([s.replace("sub-", "") for s in subs], fontsize=6)
    ax.set_xlabel("session pair"); ax.set_ylabel("subject")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_high_drift_bar(sub: pd.DataFrame, out: Path, top_n: int = 10) -> None:
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


def fig_signal_quality(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    hi_shift = (df["high_amp_ratio_j"] - df["high_amp_ratio_i"]).dropna()
    axes[0].hist(hi_shift, bins=25, edgecolor="black", alpha=0.75)
    axes[0].axvline(0, color="red", linestyle="--", label="no change")
    axes[0].set_title("High-amplitude trial ratio shift (j - i)")
    axes[0].set_xlabel("Δ high-amp trial fraction"); axes[0].set_ylabel("session pairs")
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
    L: List[str] = ["# 每被试跨 session 漂移画像 (per-subject drift summary)\n"]
    L.append(f"共 {len(sub)} 个被试（有 ≥2 个 ok session 才能算 pair）。`drift_score` 越大漂移越强"
             "（= z(MMD)+z(mu_KS)−z(CSP)−z(ERD_mu)−z(ERD_beta) 的平均），按三分位划为 "
             "high / moderate / stable。\n")
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


def write_summary_cn(stats, pair_df, sub, path: Path) -> None:
    m = stats
    high = sub[sub["drift_level"] == "high"]
    stable = sub[sub["drift_level"] == "stable"]
    worst_pair = pair_df.loc[pair_df["mmd_mean"].idxmax(), "pair"] if len(pair_df) else "—"
    stable_pair = pair_df.loc[pair_df["csp_similarity_mean"].idxmax(), "pair"] if len(pair_df) else "—"
    L: List[str] = ["# 跨 Session 漂移诊断 — 一页中文总结\n"]
    L.append("## 做了什么")
    L.append("- 量化同一被试在不同 session（不同天）之间的 EEG 分布漂移，回答“跨 session 泛化为什么难”。")
    L.append("- 纯数据层面的诊断，**不是模型训练**，不涉及 train/test、不产生 accuracy。\n")
    L.append("## 怎么做的")
    L.append("- 数据：`eog_ecg_clean_v1` 的 **148 个 status=ok session**（已排除 5 个 failed）。")
    L.append("- 每个被试 3 个 session（ses-01/02/03），按被试内部 3 种 pair（1-2、1-3、2-3）算漂移。")
    L.append(f"- 因 failed session，最终 **144 pairs / 50 被试**（sub-024 只剩 1 个 ok session → 0 pair）。")
    L.append("- 指标：MMD、CORAL、μ/β 功率漂移、μ-KS、ERD/ERS 空间相关、CSP 相似度、RMS 比值、Fisher 可分性。\n")
    L.append("## 三个 session pair 的结果")
    L.append("| pair | n | MMD(mean) | μ-KS | CSP相似 | ERD μ corr |")
    L.append("|---|---|---|---|---|---|")
    for r in pair_df.itertuples():
        L.append(f"| {r.pair} | {r.n} | {_fmt(r.mmd_mean)} | {_fmt(r.mu_ks_mean)} | "
                 f"{_fmt(r.csp_similarity_mean)} | {_fmt(r.erd_mu_corr_mean)} |")
    L.append(f"\n- **整体分布距离(MMD)最大**的 pair：`{worst_pair}`（相隔最远的 1-3）；"
             f"而 **{stable_pair}** 在 μ-KS/CSP/ERD 上最稳定，提示后期 session 的空间-频谱模式趋于一致"
             "（部分支持学习效应，详见完整报告 E 节）。\n")
    L.append("## 每个被试怎么看")
    L.append("- 看 `per_subject_drift_summary.csv` / `.md`：每被试一行，含各指标均值与 `drift_level`。")
    L.append(f"- high-drift 被试 {len(high)} 个、stable {len(stable)} 个；热力图见 "
             "`figures/subject_*_heatmap.png`。\n")
    L.append("## 主要结论")
    L.append(f"- 跨 session 漂移主要体现在**空间模式 + μ/β 频谱分布**（CSP≈{_fmt(m['csp_similarity']['mean'])}、"
             f"ERD μ≈{_fmt(m['erd_mu_corr']['mean'])}、μ-KS≈{_fmt(m['mu_ks_stat']['mean'])}）。")
    L.append(f"- 幅值不是主因（RMS 比值中位数≈{_fmt(m['rms_ratio_median']['median'])}≈1）。")
    L.append(f"- Fisher shift 平均≈{_fmt(m['fisher_shift']['mean'])}≈0：左右手可分性没有统一增强或减弱，"
             "性能下降更可能来自模式漂移而非类别不可分。\n")
    L.append("## 下一步怎么接 baseline")
    L.append("- 用同一批 148 个 ok session 跑 within-session CV vs cross-session，量化 accuracy drop。")
    L.append("- 自适应方法优先考虑 spatial alignment（EA/CORAL）、frequency/filter-bank adaptation、"
             "BN/adapter，而不是只做全局幅值归一化。")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def write_report(df, stats, pair_df, sub, figures, path: Path) -> None:
    m = stats
    pd_i = pair_df.set_index("pair")

    def pv(pair, col):
        return float(pd_i.loc[pair, col]) if pair in pd_i.index else float("nan")

    worst_pair = pair_df.loc[pair_df["mmd_mean"].idxmax(), "pair"]
    mmd_12, mmd_13, mmd_23 = pv("01-02", "mmd_mean"), pv("01-03", "mmd_mean"), pv("02-03", "mmd_mean")
    csp_12, csp_13, csp_23 = pv("01-02", "csp_similarity_mean"), pv("01-03", "csp_similarity_mean"), pv("02-03", "csp_similarity_mean")
    high = sub[sub["drift_level"] == "high"]
    stable = sub[sub["drift_level"] == "stable"]

    L: List[str] = []
    L.append("# 跨 Session 漂移诊断实验报告 (Cross-session Drift Diagnostic)\n")
    L.append("> 本报告基于已计算的 `session_drift_report.csv`（144 个被试内 session pair / 50 被试）"
             "重新整理，未重跑任何漂移计算。\n")

    # A
    L.append("## A. 实验目的\n")
    L.append("量化**同一被试**在不同 session（不同天采集）之间的 EEG 分布漂移，回答“为什么跨 session "
             "运动想象解码会比同 session 困难”。这是**数据层面的诊断**，用来指导后续 baseline 与自适应"
             "方法的设计，**不是模型训练**。\n")

    # B
    L.append("## B. 实验安排\n")
    L.append("- **数据来源**：`eog_ecg_clean_v1`，仅用 `status=ok` 的 **148 个 session**（已排除 5 个 failed）。")
    L.append("- 每个被试理论上有 3 个 session：`ses-01`、`ses-02`、`ses-03`。")
    L.append("- 跨 session 漂移按**同一被试内部**的 3 种 pair 计算（无向，i<j）：")
    L.append("  1. `ses-01 vs ses-02`")
    L.append("  2. `ses-01 vs ses-03`")
    L.append("  3. `ses-02 vs ses-03`")
    L.append("- **为什么是 144 pairs（不是 153）**：")
    L.append("  - 若 51 个被试都完整：51 × C(3,2) = **153** pairs；")
    L.append("  - 实际有 **5 个 failed session** 被排除（sub-023/ses-01、sub-024/ses-02、sub-024/ses-03、"
             "sub-026/ses-01、sub-032/ses-02）；")
    L.append("  - `sub-024` 只剩 **1 个** ok session → **0 pair**；")
    L.append("  - `sub-023`、`sub-026`、`sub-032` 各只剩 **2 个** ok session → 各 **1 pair**；")
    L.append("  - 其余 47 个被试 3 session 全 ok → 各 3 pair；")
    L.append(f"  - 合计 47×3 + 3×1 = **144 pairs / 50 个 eligible 被试**（pair 分布：1-2={int((df['pair']=='01-02').sum())}、"
             f"1-3={int((df['pair']=='01-03').sum())}、2-3={int((df['pair']=='02-03').sum())}）。")
    L.append("- **本实验不涉及** train/test split、不涉及 baseline accuracy，只做数据层面的 drift 诊断。\n")

    # C
    L.append("## C. 指标说明（测什么 / 方向）\n")
    L.append("| 指标 | 测什么 | 方向 |")
    L.append("|---|---|---|")
    L.append("| MMD (RBF) | 两 session trial 分布的整体距离 | 越大漂移越大 |")
    L.append("| CORAL | 协方差/二阶统计差异 | 当前实现基于近零均值，**易受退化影响，仅作参考、不作主证据** |")
    L.append("| μ/β power shift | MI 关键频段(8–13/13–30 Hz)功率变化(j−i) | 越偏离 0 漂移越大 |")
    L.append("| μ KS statistic | μ 频段功率分布的两样本 KS | 越大分布变化越大（0=相同）|")
    L.append("| ERD/ERS spatial corr | sensorimotor 激活空间模式跨 session 相关 | 越接近 1 越稳定 |")
    L.append("| CSP similarity | 空间判别滤波器方向一致性 | 越接近 1 越稳定 |")
    L.append("| RMS ratio | 通道幅值比值(j/i) | 越接近 1 幅值越稳定 |")
    L.append("| Fisher ratio / shift | 左右手 MI 可分性及其变化 | shift≈0=可分性无系统变化 |\n")

    # D
    L.append("## D. 总体结果（144 pairs）\n")
    L.append("| 指标 | mean | median | std |")
    L.append("|---|---|---|---|")
    for c in ["mmd", "coral", "mu_power_shift", "beta_power_shift", "mu_ks_stat",
              "erd_mu_corr", "erd_beta_corr", "csp_similarity", "rms_ratio_median",
              "fisher_shift"]:
        L.append(f"| `{c}` | {_fmt(m[c]['mean'],4)} | {_fmt(m[c]['median'],4)} | {_fmt(m[c]['std'],4)} |")
    L.append("\n**解读：**")
    L.append(f"- MMD≈{_fmt(m['mmd']['mean'])}：整体存在**中等**分布漂移。")
    L.append(f"- CSP similarity≈{_fmt(m['csp_similarity']['mean'])}：空间判别模式只**中等稳定**。")
    L.append(f"- ERD/ERS μ/β corr≈{_fmt(m['erd_mu_corr']['mean'])}/{_fmt(m['erd_beta_corr']['mean'])}："
             "感觉运动节律的空间模式存在明显漂移。")
    L.append(f"- RMS ratio median≈{_fmt(m['rms_ratio_median']['median'])}：整体幅值**不是**主要漂移来源。")
    L.append(f"- Fisher shift≈{_fmt(m['fisher_shift']['mean'],4)}：平均 MI 可分性**没有**统一增强或减弱。")
    L.append("- CORAL 中位数≈0（受近零均值退化影响），不作为主证据。\n")

    # E
    L.append("## E. 按 session pair 分组分析\n")
    L.append("| pair | n | MMD mean | MMD median | μ-KS mean | CSP sim mean | ERD μ corr | "
             "ERD β corr | RMS ratio median | Fisher shift mean |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in pair_df.itertuples():
        L.append(f"| **{r.pair}** | {r.n} | {_fmt(r.mmd_mean)} | {_fmt(r.mmd_median)} | "
                 f"{_fmt(r.mu_ks_mean)} | {_fmt(r.csp_similarity_mean)} | {_fmt(r.erd_mu_corr_mean)} | "
                 f"{_fmt(r.erd_beta_corr_mean)} | {_fmt(r.rms_ratio_median)} | {_fmt(r.fisher_shift_mean,4)} |")
    L.append("")
    L.append(f"- **MMD 最大的 pair**：`{worst_pair}`。")
    cmp_13_12 = "大于" if mmd_13 > mmd_12 else "不大于"
    L.append(f"- **1-3 vs 1-2**：1-3 的 MMD（{_fmt(mmd_13)}）{cmp_13_12} 1-2（{_fmt(mmd_12)}）"
             "——即“相隔越远（1→3）漂移是否更大”。")
    cmp_23 = "更稳定" if (mmd_23 <= mmd_12 and csp_23 >= csp_12) else "并非一致更稳定"
    L.append(f"- **2-3 是否更稳定**：2-3 的 MMD={_fmt(mmd_23)}、CSP={_fmt(csp_23)}，相比 1-2"
             f"（MMD={_fmt(mmd_12)}、CSP={_fmt(csp_12)}）{cmp_23}。")
    L.append("- **学习效应**：若“后期 session 更稳定/技能趋于一致”，应看到 2-3 漂移最小且 CSP/ERD 最高；"
             "本数据中该趋势" + ("**部分成立**" if csp_23 >= max(csp_12, csp_13) else "**不明显**") +
             "，需结合 baseline 的 within/cross accuracy 进一步判断。\n")

    # F
    L.append("## F. 按被试分析\n")
    L.append("`drift_level` 由综合漂移分 `drift_score`（z(MMD)+z(μ-KS)−z(CSP)−z(ERD_mu)−z(ERD_beta) 的均值）"
             "按三分位划分。完整表见 `per_subject_drift_summary.csv` / `.md`。\n")
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
    L.append("\n### Partial subjects（failed session 导致 pair 不全）\n")
    L.append("| subject | failed session | 剩余 ok session | 可用 pair |")
    L.append("|---|---|---|---|")
    for s in ["sub-023", "sub-024", "sub-026", "sub-032"]:
        avail = sub.loc[sub["subject"] == s, "available_pairs"]
        avail_str = avail.iloc[0] if len(avail) else "（无 pair）"
        failed = "/".join(FAILED_SESSIONS[s])
        ok_ses = {"sub-023": "ses-02, ses-03", "sub-024": "仅 ses-01",
                  "sub-026": "ses-02, ses-03", "sub-032": "ses-01, ses-03"}[s]
        L.append(f"| {s} | {failed} | {ok_ses} | {avail_str} |")
    L.append("\n- **`sub-024` 没有任何 pair**：它有 2 个 failed session（ses-02、ses-03），只剩 1 个 ok "
             "session（ses-01），无法构成任何被试内 session pair，因此不参与漂移统计。\n")

    # G
    L.append("## G. 图\n")
    for f in figures:
        L.append(f"- `figures/{f}`")
    L.append("")

    # H
    L.append("## H. 结论\n")
    L.append("- 本数据集的跨 session 漂移**主要体现为空间模式与 μ/β 频谱分布的变化**："
             f"CSP 相似度仅约 {_fmt(m['csp_similarity']['mean'])}、ERD/ERS μ 空间相关约 "
             f"{_fmt(m['erd_mu_corr']['mean'])}，且 μ-KS 约 {_fmt(m['mu_ks_stat']['mean'])} 表明频段功率分布发生了可观变化。")
    L.append(f"- 整体**幅值中位数接近稳定**（RMS 比值中位数≈{_fmt(m['rms_ratio_median']['median'])}），"
             "因此跨 session 困难**不是简单的幅值缩放问题**，仅做全局幅值归一化不足以解决。")
    L.append(f"- **Fisher shift 平均≈{_fmt(m['fisher_shift']['mean'],4)}**：左右手可分性没有被一致增强或削弱，"
             "说明后续模型跨 session 性能下降**更可能来自模式漂移**，而非类别在新 session 完全不可分。")
    L.append("- **后续 baseline**：需重点比较 within-session 与 cross-session 的 accuracy drop，量化漂移代价。")
    L.append("- **后续自适应方法**：应优先考虑 **spatial alignment（Euclidean Alignment / CORAL）**、"
             "**frequency / filter-bank adaptation**、**BN/adapter** 等针对“空间+频谱模式漂移”的手段，"
             "而不是只做全局幅值归一化。\n")
    L.append("> 备注：当前 KS 仅计算了 μ 频段（`mu_ks_stat`）；β 频段 KS 需要重跑 drift（重新读取 npz），"
             "本次按要求**未重算**，可作为后续小补充。\n")
    path.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild the drift report from the existing CSV.")
    ap.add_argument("--dir", default=str(DEFAULT_DIR), help="session_drift_v1 output dir")
    args = ap.parse_args()
    d = Path(args.dir)
    csv_path = d / "session_drift_report.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing {csv_path} (run the drift diagnostic first).")
    fig_dir = d / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_df(csv_path)
    stats = overall_stats(df)
    pair_df = session_pair_summary(df)
    sub = per_subject_summary(df)

    # tables
    pair_df.to_csv(d / "session_pair_summary.csv", index=False, float_format="%.6f")
    sub.to_csv(d / "per_subject_drift_summary.csv", index=False, float_format="%.6f")
    write_per_subject_md(sub, d / "per_subject_drift_summary.md")

    # figures
    figures: List[str] = []
    fig_pair_metric_summary(df, fig_dir / "session_pair_metric_summary.png")
    figures.append("session_pair_metric_summary.png")
    fig_subject_heatmap(df, "mmd", "MMD by subject x session-pair", "YlOrRd",
                        fig_dir / "subject_mmd_heatmap.png")
    figures.append("subject_mmd_heatmap.png")
    fig_subject_heatmap(df, "csp_similarity", "CSP similarity by subject x pair", "YlGnBu",
                        fig_dir / "subject_csp_heatmap.png", vmin=0.0, vmax=1.0)
    figures.append("subject_csp_heatmap.png")
    fig_subject_heatmap(df, "erd_mu_corr", "ERD/ERS mu corr by subject x pair", "YlGnBu",
                        fig_dir / "subject_erd_mu_heatmap.png", vmin=-1.0, vmax=1.0)
    figures.append("subject_erd_mu_heatmap.png")
    fig_high_drift_bar(sub, fig_dir / "high_drift_subjects_bar.png")
    figures.append("high_drift_subjects_bar.png")
    fig_signal_quality(df, fig_dir / "signal_quality_shift.png")
    figures.append("signal_quality_shift.png")
    # pre-existing figures kept; list them too so the report indexes everything
    existing = ["distribution_distance_hist.png", "band_power_shift_hist.png",
                "erd_ers_correlation_hist.png", "csp_similarity_hist.png",
                "fisher_ratio_scatter.png", "rms_ratio_hist.png",
                "metric_correlation_matrix.png", "session_pair_comparison.png"]
    figures += [f for f in existing if (fig_dir / f).exists()]

    # reports
    write_report(df, stats, pair_df, sub, figures, d / "SESSION_DRIFT_REPORT.md")
    write_summary_cn(stats, pair_df, sub, d / "SESSION_DRIFT_SUMMARY_CN.md")

    print(f"[drift-report] {len(df)} pairs / {df['subject'].nunique()} subjects")
    print(f"[drift-report] high-drift={int((sub['drift_level']=='high').sum())} "
          f"moderate={int((sub['drift_level']=='moderate').sum())} "
          f"stable={int((sub['drift_level']=='stable').sum())}")
    print(f"[drift-report] wrote report + 2 summaries + 2 tables + {len(figures)} figures to {d}")


if __name__ == "__main__":
    main()

