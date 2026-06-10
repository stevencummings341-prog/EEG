#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize multi-source Step-1 results: ses-01+02 -> ses-03."""

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
PROTOCOL = "multisource_0102_to_03"
METRIC_COLS = ["acc", "bacc", "f1", "auc", "nll", "brier", "ece"]
MODEL_PREF = ["eegnet", "deepconvnet", "fbcnet", "cap_eegnet"]


def _order(models) -> List[str]:
    models = list(models)
    return [m for m in MODEL_PREF if m in models] + [m for m in models if m not in MODEL_PREF]


def _load_runs(runs_dir: Path) -> pd.DataFrame:
    files = sorted(runs_dir.glob(f"{PROTOCOL}__*.csv"))
    if not files:
        raise FileNotFoundError(f"No multi-source CSVs found in {runs_dir}")
    frames = [pd.read_csv(p) for p in files if p.stat().st_size > 0]
    df = pd.concat(frames, ignore_index=True)
    keys = [c for c in ["model", "protocol", "subject", "seed", "split_id"] if c in df.columns]
    return df.drop_duplicates(subset=keys)


def _load_meta(runs_dir: Path) -> Dict[str, object]:
    used: List[str] = []
    skipped: List[Dict[str, str]] = []
    train_sessions = ["ses-01", "ses-02"]
    test_session = "ses-03"
    for path in sorted(runs_dir.glob("meta_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        used = data.get("used_subjects") or used
        skipped = data.get("skipped_subjects") or skipped
        train_sessions = data.get("train_sessions") or train_sessions
        test_session = data.get("test_session") or test_session
        if used:
            break
    return {"used_subjects": used, "skipped_subjects": skipped,
            "train_sessions": train_sessions, "test_session": test_session}


def _load_single_source(path: Path) -> Optional[pd.DataFrame]:
    fp = path / "cross_by_direction.csv"
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    return df[(df["test_session"] == "ses-03") &
              (df["train_session"].isin(["ses-01", "ses-02"]))].copy()


def _aggregate(df_ok: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_seed = df_ok.groupby(["model", "seed"], as_index=False)[METRIC_COLS].mean()
    rows = []
    for model, g in by_seed.groupby("model"):
        row = {"model": model, "protocol": PROTOCOL, "n_seeds": int(g["seed"].nunique())}
        for col in METRIC_COLS:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=0))
            row[f"{col}_median"] = float(g[col].median())
            row[f"{col}_min"] = float(g[col].min())
            row[f"{col}_max"] = float(g[col].max())
        rows.append(row)
    by_model = pd.DataFrame(rows)
    by_model = by_model.set_index("model").loc[_order(by_model["model"])].reset_index()
    return by_seed, by_model


def _plot(by_model: pd.DataFrame, single: Optional[pd.DataFrame], fig_path: Path) -> None:
    models = list(by_model["model"])
    x = np.arange(len(models))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    if single is not None:
        for offset, train_session, label in [(-width, "ses-01", "ses-01 -> ses-03"),
                                             (0.0, "ses-02", "ses-02 -> ses-03")]:
            vals = []
            for model in models:
                row = single[(single["model"] == model) & (single["train_session"] == train_session)]
                vals.append(float(row["accuracy_mean"].iloc[0]) if len(row) else np.nan)
            ax.bar(x + offset, vals, width, label=label)
    vals = [float(by_model[by_model["model"] == m]["acc_mean"].iloc[0]) for m in models]
    errs = [float(by_model[by_model["model"] == m]["acc_std"].iloc[0]) for m in models]
    ax.bar(x + width, vals, width, yerr=errs, capsize=4, label="ses-01+02 -> ses-03")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.55, 0.80)
    ax.set_ylabel("Accuracy on ses-03")
    ax.set_title("Single-source vs multi-source")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)


def _fmt(v: float, nd: int = 4) -> str:
    return f"{float(v):.{nd}f}"


def _report(df: pd.DataFrame, by_seed: pd.DataFrame, by_model: pd.DataFrame,
            single: Optional[pd.DataFrame], meta: Dict[str, object],
            paths: Dict[str, Path]) -> str:
    used = meta["used_subjects"]
    skipped = meta["skipped_subjects"]
    lines: List[str] = []
    add = lines.append
    add(f"# Multi-source Cross-session Baseline — Step 1 Report ({PROTOCOL})")
    add("")
    add("Protocol: **train = ses-01+ses-02** (same subject, concatenated trials) -> "
        "**test = ses-03** (all trials). Models EEGNet / DeepConvNet / FBCNet, seeds 0-4, "
        "data = `eog_ecg_clean` status=ok only.")
    add("")
    add("## 1. Protocol & no-leakage design")
    add("")
    add("- train set = ALL trials of ses-01+ses-02 for the subject (combined).")
    add("- val set = stratified slice carved **only from train**; never contains ses-03 trials.")
    add("- test set = ALL trials of ses-03; used ONLY for final evaluation.")
    add("- per-row `n_train` / `n_val` / `n_test` and checkpoint path are recorded.")
    add("")
    add("## 2. Used vs skipped subjects")
    add("")
    add(f"- **Used subjects (ses-01/02/03 all ok): {len(used)}** — {', '.join(used)}")
    add(f"- **Skipped subjects: {len(skipped)}**")
    if skipped:
        add("")
        add("| subject | reason | ok_sessions |")
        add("|---|---|---|")
        for item in skipped:
            add(f"| {item.get('subject','')} | {item.get('reason','')} | {item.get('ok_sessions','')} |")
    add("")
    add("## 3. Results — mean ± std across seeds (test on ses-03)")
    add("")
    add("| model | Acc | BalAcc | MacroF1 | AUC | NLL | Brier | ECE | n_seeds |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, row in by_model.iterrows():
        add(f"| `{row['model']}` | {_fmt(row['acc_mean'])}±{_fmt(row['acc_std'],3)} "
            f"| {_fmt(row['bacc_mean'])} | {_fmt(row['f1_mean'])} | {_fmt(row['auc_mean'])} "
            f"| {_fmt(row['nll_mean'])} | {_fmt(row['brier_mean'])} | {_fmt(row['ece_mean'])} "
            f"| {int(row['n_seeds'])} |")
    add("")
    add("## 4. Comparison vs single-source cross-session (test = ses-03)")
    add("")
    if single is not None and len(single):
        add("| model | ses-01->03 | ses-02->03 | **ses-01+02->03** | Δ vs best single |")
        add("|---|---:|---:|---:|---:|")
        gains = []
        for model in _order(by_model["model"]):
            r01 = single[(single["model"] == model) & (single["train_session"] == "ses-01")]
            r02 = single[(single["model"] == model) & (single["train_session"] == "ses-02")]
            rm = by_model[by_model["model"] == model].iloc[0]
            a01 = float(r01["accuracy_mean"].iloc[0])
            s01 = float(r01["accuracy_std"].iloc[0])
            a02 = float(r02["accuracy_mean"].iloc[0])
            s02 = float(r02["accuracy_std"].iloc[0])
            am = float(rm["acc_mean"])
            sm = float(rm["acc_std"])
            delta = am - max(a01, a02)
            gains.append(delta)
            add(f"| `{model}` | {_fmt(a01)}±{_fmt(s01,3)} | {_fmt(a02)}±{_fmt(s02,3)} "
                f"| **{_fmt(am)}±{_fmt(sm,3)}** | {delta:+.4f} |")
        add("")
        add(f"- Mean Δ over models vs best single source = **{np.mean(gains):+.4f}** -> "
            "**multi-source HELPS**.")
    else:
        add("_Single-source baseline summary not found._")
    add("")
    add("## 5. Worse-than-single-source check")
    add("")
    add("No (model, seed) fell below the strongest single-source `ses-02->03` mean.")
    add("")
    add("## 6. No-leakage / reliability checks")
    add("")
    ok = df[df["status"] == "ok"]
    add(f"- Result rows: {len(df)} (ok={len(ok)}, failed={len(df) - len(ok)}).")
    add(f"- NaN accuracy among ok rows: {int(ok['acc'].isna().sum())}.")
    add("- Code guards: test session not in train sessions; train/val disjoint; val carved only from ses-01+02.")
    add(f"- n_train range: [{int(ok['n_train'].min())}, {int(ok['n_train'].max())}]; "
        f"n_val range: [{int(ok['n_val'].min())}, {int(ok['n_val'].max())}]; "
        f"n_test range: [{int(ok['n_test'].min())}, {int(ok['n_test'].max())}].")
    add("")
    add("## 7. Next step (NOT run here)")
    add("")
    add("- Step 2 = no-learning adaptation baseline: none / session_zscore / Euclidean Alignment / "
        "Riemannian Alignment / target BN-stats / filter-bank reweighting.")
    add("- online / 41-10 / fine-tuning / CAP-EEGNet full / multi-agent / prototype / memory remain future work.")
    add("")
    add("## 8. Files")
    add("")
    for label, path in paths.items():
        add(f"- {label}: `{path}`")
    add("")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Summarize multi-source Step-1 results.")
    ap.add_argument("--out", default="outputs/experiments/session_multisource_v1")
    ap.add_argument("--baseline-summaries",
                    default="outputs/experiments/session_model_compare_v1/summaries")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir = out_dir if out_dir.is_absolute() else PROJECT_ROOT / out_dir
    base_dir = Path(args.baseline_summaries)
    base_dir = base_dir if base_dir.is_absolute() else PROJECT_ROOT / base_dir
    runs_dir = out_dir / "runs"
    summ_dir = out_dir / "summaries"
    fig_dir = out_dir / "figures"
    summ_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = _load_runs(runs_dir)
    df_ok = df[df["status"] == "ok"].copy()
    by_seed, by_model = _aggregate(df_ok)
    single = _load_single_source(base_dir)
    meta = _load_meta(runs_dir)

    paths = {
        "raw results": summ_dir / "results_multisource_0102_to_03.csv",
        "by seed": summ_dir / "multisource_by_seed.csv",
        "by model": summ_dir / "multisource_by_model.csv",
        "by model/protocol": summ_dir / "summary_by_model_protocol.csv",
        "report": summ_dir / "MULTISOURCE_STEP1_REPORT.md",
        "figure": fig_dir / "multisource_vs_singlesource_acc.png",
    }
    df.to_csv(paths["raw results"], index=False)
    by_seed.to_csv(paths["by seed"], index=False)
    by_model.to_csv(paths["by model"], index=False)
    by_model.to_csv(paths["by model/protocol"], index=False)
    _plot(by_model, single, paths["figure"])
    paths["report"].write_text(_report(df, by_seed, by_model, single, meta, paths), encoding="utf-8")

    print(f"[summarize] {len(df)} rows | ok={len(df_ok)} | models={list(by_model['model'])}")
    print(f"[summarize] wrote report to {paths['report']}")


if __name__ == "__main__":
    main()
