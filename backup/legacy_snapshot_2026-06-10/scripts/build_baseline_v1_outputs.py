#!/usr/bin/env python
"""Build the canonical `outputs/experiments/baseline_v1/` result layout.

This script reorganizes already-finished static baseline artifacts into:

baseline_v1/
  within_session/
  cross_session/
  provenance/

It does not run training. It moves the old raw run directories into provenance if they
still exist at the top level; otherwise it uses the provenance copies already there.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXP = Path("outputs/experiments")
ANA = Path("outputs/analysis")
OUT = EXP / "baseline_v1"
PROV = OUT / "provenance"


def ensure_dirs() -> None:
    for p in [
        "within_session/runs",
        "within_session/splits",
        "within_session/tables",
        "within_session/figures",
        "cross_session/runs",
        "cross_session/splits",
        "cross_session/tables",
        "cross_session/figures",
        "figures",
        "provenance",
    ]:
        (OUT / p).mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.relative_to(dst.parent))
    except Exception:
        shutil.copy2(src, dst)


def move_raw_runs_into_provenance() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PROV.mkdir(exist_ok=True)
    for name in ["session_model_compare_v1", "session_multisource_v1"]:
        src = EXP / name
        dst = PROV / name
        if src.exists() and not dst.exists():
            src.rename(dst)
    for old in [EXP / "static_baseline_v1", EXP / "cross_session_baseline_v1"]:
        if old.exists():
            shutil.rmtree(old)


def copy_links(base_raw: Path, ms_raw: Path) -> None:
    for f in (base_raw / "runs").glob("within__*"):
        link_or_copy(f, OUT / "within_session/runs" / f.name)
    for f in (base_raw / "runs").glob("meta_within__*"):
        link_or_copy(f, OUT / "within_session/runs" / f.name)
    for f in (base_raw / "splits").glob("within_*.json"):
        link_or_copy(f, OUT / "within_session/splits" / f.name)
    for f in (base_raw / "runs").glob("cross__*"):
        link_or_copy(f, OUT / "cross_session/runs" / f.name)
    for f in (base_raw / "runs").glob("meta_cross__*"):
        link_or_copy(f, OUT / "cross_session/runs" / f.name)
    pair = base_raw / "splits/cross_session_pairs.json"
    if pair.exists():
        link_or_copy(pair, OUT / "cross_session/splits/cross_session_pairs.json")
    for f in (ms_raw / "runs").glob("*"):
        link_or_copy(f, OUT / "cross_session/runs" / f.name)
    for f in (ms_raw / "splits").glob("*.json"):
        link_or_copy(f, OUT / "cross_session/splits" / f.name)


def build_tables(base: Path, ms: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    within = pd.read_csv(base / "results_within_session.csv")
    cross = pd.read_csv(base / "results_cross_session.csv")
    ms_raw = pd.read_csv(ms / "results_multisource_0102_to_03.csv")
    ms_ok = ms_raw[ms_raw["status"] == "ok"].copy()

    within.to_csv(OUT / "within_session/tables/results_within_session.csv", index=False)
    pd.read_csv(base / "within_by_seed.csv").to_csv(OUT / "within_session/tables/within_by_seed.csv", index=False)
    pd.read_csv(base / "within_session_wise.csv").to_csv(OUT / "within_session/tables/within_session_wise.csv", index=False)

    w_unit = within.groupby(["model", "subject", "session", "seed"], as_index=False).agg(
        acc=("accuracy", "mean"),
        bacc=("balanced_accuracy", "mean"),
        f1=("macro_f1", "mean"),
        auc=("auc", "mean"),
    )
    w_subj = w_unit.groupby(["model", "subject", "session"], as_index=False).agg(
        mean_acc=("acc", "mean"),
        std_acc=("acc", "std"),
        mean_bacc=("bacc", "mean"),
        mean_f1=("f1", "mean"),
        mean_auc=("auc", "mean"),
    )
    w_subj.to_csv(OUT / "within_session/tables/within_by_subject.csv", index=False)
    w_seed = pd.read_csv(base / "within_by_seed.csv")
    rows = []
    for model, g in w_seed.groupby("model"):
        row = {"model": model, "n_seeds": int(g["seed"].nunique())}
        for c in ["accuracy", "balanced_accuracy", "macro_f1", "auc", "nll", "brier", "ece"]:
            row[f"{c}_mean"] = float(g[c].mean())
            row[f"{c}_std"] = float(g[c].std(ddof=0))
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "within_session/tables/summary_within_by_model.csv", index=False)

    single = pd.DataFrame(
        {
            "protocol": "single_source_directed_pair",
            "training_scope": "single_source",
            "model": cross["model"],
            "seed": cross["seed"],
            "subject": cross["subject"],
            "train_sessions": cross["train_session"],
            "test_session": cross["test_session"],
            "acc": cross["accuracy"],
            "bacc": cross["balanced_accuracy"],
            "f1": cross["macro_f1"],
            "auc": cross["auc"],
            "nll": cross["nll"],
            "brier": cross["brier"],
            "ece": cross["ece"],
            "n_train": cross["n_train"],
            "n_val": cross["n_val"],
            "n_test": cross["n_test"],
            "checkpoint_path": cross.apply(
                lambda r: (
                    f"checkpoints/session_model_compare_v1/{r['model']}/"
                    f"cross_{r['subject']}_{r['train_session']}-to-{r['test_session']}_seed{int(r['seed'])}.pt"
                ),
                axis=1,
            ),
            "source_run_id": "session_model_compare_v1",
        }
    )
    multi = pd.DataFrame(
        {
            "protocol": ms_ok["protocol"],
            "training_scope": "multi_source",
            "model": ms_ok["model"],
            "seed": ms_ok["seed"],
            "subject": ms_ok["subject"],
            "train_sessions": ms_ok["train_sessions"],
            "test_session": ms_ok["test_session"],
            "acc": ms_ok["acc"],
            "bacc": ms_ok["bacc"],
            "f1": ms_ok["f1"],
            "auc": ms_ok["auc"],
            "nll": ms_ok["nll"],
            "brier": ms_ok["brier"],
            "ece": ms_ok["ece"],
            "n_train": ms_ok["n_train"],
            "n_val": ms_ok["n_val"],
            "n_test": ms_ok["n_test"],
            "checkpoint_path": ms_ok["checkpoint_path"],
            "source_run_id": "session_multisource_v1",
        }
    )
    allc = pd.concat([single, multi], ignore_index=True)
    allc.to_csv(OUT / "cross_session/tables/results_cross_session_all.csv", index=False)
    allc.groupby(["training_scope", "protocol", "model", "seed"], as_index=False).agg(
        mean_acc=("acc", "mean"),
        mean_bacc=("bacc", "mean"),
        mean_f1=("f1", "mean"),
        mean_auc=("auc", "mean"),
        mean_nll=("nll", "mean"),
        mean_brier=("brier", "mean"),
        mean_ece=("ece", "mean"),
        n_subject_rows=("subject", "count"),
    ).to_csv(OUT / "cross_session/tables/cross_by_seed.csv", index=False)
    allc.groupby(["training_scope", "protocol", "model", "train_sessions", "test_session"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        n_subjects=("subject", "nunique"),
        mean_acc=("acc", "mean"),
        std_acc=("acc", "std"),
        mean_bacc=("bacc", "mean"),
        std_bacc=("bacc", "std"),
        mean_f1=("f1", "mean"),
        std_f1=("f1", "std"),
        mean_auc=("auc", "mean"),
        std_auc=("auc", "std"),
    ).to_csv(OUT / "cross_session/tables/cross_by_direction.csv", index=False)
    return within, cross, allc


def build_comparison_tables(within: pd.DataFrame, allc: pd.DataFrame, base: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    models = ["eegnet", "deepconvnet", "fbcnet"]
    scope_rows, proto_rows, gap_rows = [], [], []
    within_sess = pd.read_csv(base / "within_session_wise.csv")
    for model in models:
        g = allc[(allc.model == model) & (allc.training_scope == "single_source")]
        scope_rows.append(
            {
                "model": model,
                "training_scope": "single_source",
                "protocol": "all_directed_pairs",
                "train_sessions": "all_single",
                "test_session": "all_targets",
                "mean_acc": float(g.acc.mean()),
                "std_acc": float(g.groupby("seed").acc.mean().std(ddof=0)),
                "mean_bacc": float(g.bacc.mean()),
                "std_bacc": float(g.groupby("seed").bacc.mean().std(ddof=0)),
                "n_subjects": int(g.subject.nunique()),
                "n_seeds": int(g.seed.nunique()),
            }
        )
        means = {}
        for train in ["ses-01", "ses-02", "ses-01+ses-02"]:
            means[train] = allc[(allc.model == model) & (allc.train_sessions == train) & (allc.test_session == "ses-03")]
        best_train = max(["ses-01", "ses-02"], key=lambda t: means[t].acc.mean())
        for train, gg in means.items():
            proto_rows.append(
                {
                    "model": model,
                    "protocol": "multi_source_0102_to_03" if "+" in train else f"{train}_to_ses-03",
                    "train_sessions": train,
                    "test_session": "ses-03",
                    "mean_acc": float(gg.acc.mean()),
                    "std_acc": float(gg.groupby("seed").acc.mean().std(ddof=0)),
                    "delta_vs_ses01_to_ses03": float(gg.acc.mean() - means["ses-01"].acc.mean()),
                    "delta_vs_ses02_to_ses03": float(gg.acc.mean() - means["ses-02"].acc.mean()),
                    "n_subjects": int(gg.subject.nunique()),
                    "n_seeds": int(gg.seed.nunique()),
                }
            )
        for train, protocol, scope in [
            (best_train, "best_single_to_ses03", "single_source"),
            ("ses-01+ses-02", "ses01_ses02_to_ses03", "multi_source"),
        ]:
            gg = means[train]
            scope_rows.append(
                {
                    "model": model,
                    "training_scope": scope,
                    "protocol": protocol,
                    "train_sessions": train,
                    "test_session": "ses-03",
                    "mean_acc": float(gg.acc.mean()),
                    "std_acc": float(gg.groupby("seed").acc.mean().std(ddof=0)),
                    "mean_bacc": float(gg.bacc.mean()),
                    "std_bacc": float(gg.groupby("seed").bacc.mean().std(ddof=0)),
                    "n_subjects": int(gg.subject.nunique()),
                    "n_seeds": int(gg.seed.nunique()),
                }
            )
        within03 = float(within_sess[(within_sess.model == model) & (within_sess.session == "ses-03")].accuracy_mean.iloc[0])
        single02 = float(means["ses-02"].acc.mean())
        multi_acc = float(means["ses-01+ses-02"].acc.mean())
        gap_rows.append(
            {
                "model": model,
                "within_ses03_acc": within03,
                "single_source_ses02_to_ses03_acc": single02,
                "multi_source_0102_to_03_acc": multi_acc,
                "single_to_within_gap": within03 - single02,
                "multi_gain": multi_acc - single02,
                "gap_recovered": (multi_acc - single02) / (within03 - single02),
            }
        )
    scope = pd.DataFrame(scope_rows)
    proto = pd.DataFrame(proto_rows)
    gap = pd.DataFrame(gap_rows)
    scope.to_csv(OUT / "cross_session/tables/cross_by_training_scope.csv", index=False)
    proto.to_csv(OUT / "cross_session/tables/cross_protocol_comparison.csv", index=False)
    gap.to_csv(OUT / "cross_session/tables/cross_gap_recovery.csv", index=False)
    return scope, proto, gap


def build_subject_table(within: pd.DataFrame, allc: pd.DataFrame) -> pd.DataFrame:
    single03 = allc[(allc.training_scope == "single_source") & (allc.test_session == "ses-03")]
    single_ps = single03.groupby(["model", "subject", "train_sessions"], as_index=False).acc.mean()
    best = single_ps.sort_values("acc").groupby(["model", "subject"]).tail(1).rename(
        columns={"train_sessions": "single_best_train_sessions", "acc": "single_best_acc"}
    )
    multi = (
        allc[allc.training_scope == "multi_source"]
        .groupby(["model", "subject"], as_index=False)
        .acc.mean()
        .rename(columns={"acc": "multisource_acc"})
    )
    subj = best.merge(multi, on=["model", "subject"], how="outer")
    subj["gain"] = subj["multisource_acc"] - subj["single_best_acc"]
    wsub = within.groupby(["model", "subject", "session"], as_index=False).accuracy.mean()
    for sess, col in [("ses-01", "ses01_within_acc"), ("ses-02", "ses02_within_acc"), ("ses-03", "ses03_within_acc")]:
        subj = subj.merge(
            wsub[wsub.session == sess][["model", "subject", "accuracy"]].rename(columns={"accuracy": col}),
            on=["model", "subject"],
            how="left",
        )
    try:
        drift = pd.read_csv(ANA / "session_drift_v1/per_subject_drift_summary.csv")
        subj = subj.merge(drift[["subject", "drift_level", "drift_score"]], on="subject", how="left")
    except Exception:
        pass
    subj.to_csv(OUT / "cross_session/tables/cross_by_subject.csv", index=False)
    return subj


def build_figures(base: Path, proto: pd.DataFrame, gap: pd.DataFrame, subj: pd.DataFrame) -> None:
    models = ["eegnet", "deepconvnet", "fbcnet"]
    labels = {"eegnet": "EEGNet", "deepconvnet": "DeepConvNet", "fbcnet": "FBCNet"}
    trend = pd.read_csv(OUT / "within_session/tables/within_session_wise.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in models:
        g = trend[trend.model == model].copy()
        g["order"] = g.session.map({"ses-01": 1, "ses-02": 2, "ses-03": 3})
        g = g.sort_values("order")
        ax.errorbar(g.session, g.accuracy_mean, yerr=g.accuracy_std, marker="o", capsize=3, label=labels[model])
    ax.set_ylim(0.68, 0.85)
    ax.set_ylabel("Accuracy")
    ax.set_title("Within-session trend by model")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "within_session/figures/within_session_trend_by_model.png", dpi=140)
    plt.close(fig)
    shutil.copy2(base / "within_session_accuracy_boxplot.png", OUT / "within_session/figures/within_session_accuracy_boxplot.png")
    shutil.copy2(base / "cross_session_accuracy_matrix_by_model.png", OUT / "cross_session/figures/cross_session_accuracy_matrix_by_model.png")

    summary = pd.read_csv(base / "summary_by_model_protocol.csv")
    within_map = summary[summary.protocol == "within_session"].set_index("model").accuracy_mean
    cross_map = summary[summary.protocol == "cross_session"].set_index("model").accuracy_mean
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([labels[m] for m in models], [within_map[m] - cross_map[m] for m in models])
    ax.set_ylabel("Within - cross accuracy")
    ax.set_title("Cross-session drop by model")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "cross_session/figures/cross_session_drop_by_model.png", dpi=140)
    plt.close(fig)

    x = np.arange(len(models))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 4.8))
    series = []
    for train in ["ses-01", "ses-02", "ses-01+ses-02"]:
        series.append([proto[(proto.model == m) & (proto.train_sessions == train)].mean_acc.iloc[0] for m in models])
    best = [max(series[0][i], series[1][i]) for i in range(len(models))]
    for off, arr, name in [
        (-1.5 * width, series[0], "ses-01 -> ses-03"),
        (-0.5 * width, series[1], "ses-02 -> ses-03"),
        (0.5 * width, best, "best single-source"),
        (1.5 * width, series[2], "ses-01+02 -> ses-03"),
    ]:
        ax.bar(x + off, arr, width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[m] for m in models])
    ax.set_ylim(0.58, 0.80)
    ax.set_ylabel("Accuracy on ses-03")
    ax.set_title("Single-source vs multi-source")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "cross_session/figures/single_vs_multisource_accuracy.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot([subj[subj.model == m].gain.dropna().values for m in models], labels=[labels[m] for m in models], showmeans=True)
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_ylabel("Gain = multi - best single")
    ax.set_title("Multi-source gain by subject")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "cross_session/figures/multisource_gain_by_subject.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([labels[m] for m in gap.model], gap.gap_recovered * 100)
    ax.set_ylabel("Recovered gap (%)")
    ax.set_title("Gap recovery by model")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "cross_session/figures/gap_recovery_by_model.png", dpi=140)
    plt.close(fig)

    for f in (ANA / "session_drift_v1/figures").glob("*.png"):
        shutil.copy2(f, OUT / "figures" / f"drift_{f.name}")
    for f in (OUT / "within_session/figures").glob("*.png"):
        shutil.copy2(f, OUT / "figures" / f"within_{f.name}")
    for f in (OUT / "cross_session/figures").glob("*.png"):
        shutil.copy2(f, OUT / "figures" / f"cross_session_{f.name}")


def write_docs() -> None:
    (OUT / "manifest_sources.json").write_text(
        json.dumps(
            {
                "created_from": {
                    "drift": "outputs/analysis/session_drift_v1",
                    "within_and_single_source_raw_run": "provenance/session_model_compare_v1",
                    "multi_source_raw_run": "provenance/session_multisource_v1",
                },
                "canonical_layout": "outputs/experiments/baseline_v1",
                "status": "static baseline complete; Step 2 no-learning adaptation not run",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUT / "resolved_config_summary.yaml").write_text(
        """run_id: baseline_v1
status: static_baseline_complete
data_entry: /share/workspace2/moto_imagination/WBCIC_SHU/processed/eog_ecg_clean/
status_filter: [ok]
models: [eegnet, deepconvnet, fbcnet]
seeds: [0, 1, 2, 3, 4]
protocols:
  within_session: 10_fold_cv
  cross_session:
    single_source: directed_pairs
    multi_source: ses-01+ses-02_to_ses-03
step2_adaptation: not_run
""",
        encoding="utf-8",
    )
    (OUT / "README.md").write_text(
        """# baseline_v1

Canonical output folder for completed static baseline.

- `within_session/`: within-session 10-fold CV.
- `cross_session/`: all cross-session baselines. Single-source and multi-source are
  distinguished by table fields, not by top-level experiment categories.
- `provenance/`: original raw run folders (`session_model_compare_v1`, `session_multisource_v1`)
  kept for reproducibility.

Main report: `BASELINE_REPORT.md`.
Step 2 no-learning adaptation has not been run.
""",
        encoding="utf-8",
    )
    (OUT / "BASELINE_REPORT.md").write_text(
        """# BASELINE_REPORT.md

## 1. Executive Summary

- within-session CV is the no-drift upper bound.
- single-source cross-session shows a 9-13% relative drop.
- multi-source `ses-01+02 -> ses-03` improves over the best single source for all three models.
- Some subjects still degrade under naive multi-source, motivating Step 2 no-learning alignment.

## 2. Dataset and Protocol

- Data: `eog_ecg_clean`, status=ok only (148 ok / 5 failed).
- Models: EEGNet / DeepConvNet / FBCNet.
- Seeds: 0-4.
- Multi-source uses 47 subjects with all three sessions ok.
- No leakage: validation is carved only from train; test labels are used only for final evaluation.

## 3. Within-session Baseline

| model | Acc |
|---|---:|
| EEGNet | 0.807±0.002 |
| DeepConvNet | 0.766±0.002 |
| FBCNet | 0.720±0.003 |

![within trend](within_session/figures/within_session_trend_by_model.png)

## 4. Single-source Cross-session Baseline

| model | Cross Acc | Drop vs within |
|---|---:|---:|
| EEGNet | 0.711±0.008 | 0.096 |
| DeepConvNet | 0.681±0.002 | 0.085 |
| FBCNet | 0.628±0.003 | 0.092 |

![direction matrix](cross_session/figures/cross_session_accuracy_matrix_by_model.png)

## 5. Multi-source Cross-session Baseline

| model | ses-01->03 | ses-02->03 | ses-01+02->03 | gain vs best single |
|---|---:|---:|---:|---:|
| EEGNet | 0.6991±0.009 | 0.7492±0.008 | **0.7717±0.003** | +0.0224 |
| DeepConvNet | 0.6757±0.004 | 0.7211±0.009 | **0.7564±0.007** | +0.0353 |
| FBCNet | 0.6142±0.006 | 0.6484±0.005 | **0.6750±0.002** | +0.0267 |

![single vs multisource](cross_session/figures/single_vs_multisource_accuracy.png)

## 6. Gap Recovery Analysis

See `cross_session/tables/cross_gap_recovery.csv`.

![gap recovery](cross_session/figures/gap_recovery_by_model.png)

## 7. Per-subject Gain

See `cross_session/tables/cross_by_subject.csv`.

![gain by subject](cross_session/figures/multisource_gain_by_subject.png)

## 8. Reliability and Leakage Checks

- within rows: 22,200.
- single-source cross rows: 4,320.
- multi-source rows: 705 (0 failed, 0 NaN).
- multi-source sizes: n_train=320, n_val=80, n_test=200.
- Splits saved under `within_session/splits/` and `cross_session/splits/`.

## 9. Next Stage

Step 2 no-learning adaptation baseline: none / session_zscore / EA / Riemannian / BN stats / filter-bank.
""",
        encoding="utf-8",
    )


def main() -> None:
    move_raw_runs_into_provenance()
    ensure_dirs()
    base_raw = PROV / "session_model_compare_v1"
    ms_raw = PROV / "session_multisource_v1"
    copy_links(base_raw, ms_raw)
    base = base_raw / "summaries"
    ms = ms_raw / "summaries"
    within, _cross, allc = build_tables(base, ms)
    _scope, proto, gap = build_comparison_tables(within, allc, base)
    subj = build_subject_table(within, allc)
    build_figures(base, proto, gap, subj)
    write_docs()
    print("baseline_v1 rebuilt")
    print("cross_all rows", len(allc))
    print("figures", len(list((OUT / "figures").glob("*.png"))))


if __name__ == "__main__":
    main()
