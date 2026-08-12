#!/usr/bin/env python
"""Aggregate a cross-subject run into the comparison table (+ the paper's own numbers).

Reads every ``cells/*/result.json`` of a run (the authoritative per-cell record) and writes:

  * ``tables/per_cell.csv``       — one row per (model, fold, test subject)
  * ``tables/per_model.csv``      — mean +/- std over folds, per model
  * ``tables/vs_paper.csv``       — our numbers next to the DSGNet paper's Table II (SHUv5)
  * ``REPORT_TABLE.md``           — the same comparison as Markdown, ready to paste

Paper numbers are quoted from Table II of Lou et al., IEEE JBHI 2026
(doi:10.1109/jbhi.2026.3689121), SHUv5 three-class LOSO column. They are *reference values
measured by the authors*, not something we reproduced: our split protocol matches, our
optimizer recipe matches for this run, but preprocessing entry and implementation differ.

Usage:
  python scripts/summarize_cross_subject.py --run outputs/experiments/wbci_shu/paper_baseline_3c_821_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

#: Table II, SHUv5 (three-class, LOSO, mean over subjects). Acc / macro-F1 / Kappa.
PAPER_SHUV5: Dict[str, Dict[str, float]] = {
    "eegnet_official": {"acc": 0.6492, "f1": 0.6488, "kappa": 0.4737, "as": "EEGNet [18]"},
    "eegnex": {"acc": 0.6488, "f1": 0.6469, "kappa": 0.4731, "as": "EEGNeX [20]"},
    "eeg_deformer": {"acc": 0.6529, "f1": 0.6503, "kappa": 0.4793, "as": "EEG-Deformer [23]"},
    "atcnet": {"acc": 0.6834, "f1": 0.6826, "kappa": 0.5275, "as": "ATCNet [24]"},
}
#: Models the paper reports but we do not run (no complete official code) — quoted only.
PAPER_ONLY: Dict[str, Dict[str, float]] = {
    "EEGInception [27]": {"acc": 0.5809, "f1": 0.5778, "kappa": 0.3713},
    "MDGEEG [35]": {"acc": 0.6688, "f1": 0.6634, "kappa": 0.5031},
    "EEG-DG [38]": {"acc": 0.6509, "f1": 0.6488, "kappa": 0.4763},
    "DSGNet (proposed)": {"acc": 0.6856, "f1": 0.6833, "kappa": 0.5284},
}

CELL_FIELDS = ["model", "fold", "test_subject", "n_train_subjects", "n_val_subjects",
               "n_train", "n_val", "n_test", "best_epoch", "epochs_run", "early_stopped",
               "n_params", "train_seconds", "accuracy", "balanced_accuracy", "macro_f1",
               "auc", "last_accuracy", "last_macro_f1"]


def collect(run_dir: Path, seed: int) -> List[Dict]:
    rows: List[Dict] = []
    for result_json in sorted((run_dir / "cells").glob(f"*__seed{seed}/result.json")):
        detail = json.loads(result_json.read_text(encoding="utf-8"))
        for r in detail.get("rows") or []:
            if r.get("status") != "ok":
                continue
            rows.append({k: r.get(k, "") for k in CELL_FIELDS})
    return rows


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def agg(rows: List[Dict]) -> List[Dict]:
    by_model: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    out = []
    for model, rs in by_model.items():
        acc = [float(r["accuracy"]) for r in rs]
        f1 = [float(r["macro_f1"]) for r in rs]
        out.append({
            "model": model,
            "n_folds": len(rs),
            "n_params": rs[0]["n_params"],
            "acc_mean": round(st.mean(acc), 4),
            "acc_std": round(st.pstdev(acc) if len(acc) > 1 else 0.0, 4),
            "acc_min": round(min(acc), 4),
            "acc_max": round(max(acc), 4),
            "f1_mean": round(st.mean(f1), 4),
            "f1_std": round(st.pstdev(f1) if len(f1) > 1 else 0.0, 4),
            "mean_best_epoch": round(st.mean(int(r["best_epoch"]) for r in rs), 1),
            "gpu_hours": round(sum(float(r["train_seconds"]) for r in rs) / 3600, 2),
        })
    return sorted(out, key=lambda d: -d["acc_mean"])


def fmt(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run)
    rows = collect(run_dir, args.seed)
    if not rows:
        raise SystemExit(f"no completed cells under {run_dir}/cells")

    write_csv(run_dir / "tables" / "per_cell.csv", CELL_FIELDS, rows)
    per_model = agg(rows)
    write_csv(run_dir / "tables" / "per_model.csv", list(per_model[0]), per_model)

    # ---- our numbers vs the paper's ----
    vs_rows: List[Dict] = []
    for d in per_model:
        ref = PAPER_SHUV5.get(d["model"])
        vs_rows.append({
            "model": d["model"],
            "kind": "published baseline (reproduced here)" if ref else "ours",
            "paper_name": (ref or {}).get("as", ""),
            "ours_acc": d["acc_mean"], "ours_acc_std": d["acc_std"],
            "ours_f1": d["f1_mean"],
            "paper_acc": (ref or {}).get("acc", ""),
            "paper_f1": (ref or {}).get("f1", ""),
            "delta_acc": (round(d["acc_mean"] - ref["acc"], 4) if ref else ""),
            "n_folds": d["n_folds"],
        })
    for name, ref in PAPER_ONLY.items():
        vs_rows.append({
            "model": name, "kind": "paper number only (no official code to run)",
            "paper_name": name, "ours_acc": "", "ours_acc_std": "", "ours_f1": "",
            "paper_acc": ref["acc"], "paper_f1": ref["f1"], "delta_acc": "", "n_folds": "",
        })
    write_csv(run_dir / "tables" / "vs_paper.csv", list(vs_rows[0]), vs_rows)

    # ---- markdown ----
    md = [f"# {run_dir.name} — cross-subject comparison", "",
          f"Cells aggregated: {len(rows)} (seed {args.seed}).", "",
          "## Our run", "",
          "| model | Acc | macro-F1 | folds | params | best ep | GPU-h |",
          "|:---|---:|---:|---:|---:|---:|---:|"]
    for d in per_model:
        md.append(f"| `{d['model']}` | **{d['acc_mean']:.4f}** ±{d['acc_std']:.4f} | "
                  f"{d['f1_mean']:.4f} ±{d['f1_std']:.4f} | {d['n_folds']} | "
                  f"{int(d['n_params']):,} | {d['mean_best_epoch']} | {d['gpu_hours']} |")
    md += ["", "## Reproduced baselines vs the paper (SHUv5 3C, LOSO)", "",
           "| model | ours Acc | paper Acc | Δ | ours F1 | paper F1 |",
           "|:---|---:|---:|---:|---:|---:|"]
    for r in vs_rows:
        if r["kind"].startswith("published"):
            md.append(f"| {r['paper_name']} | {r['ours_acc']:.4f} | {r['paper_acc']:.4f} | "
                      f"{r['delta_acc']:+.4f} | {r['ours_f1']:.4f} | {r['paper_f1']:.4f} |")
    md += ["", "## Paper numbers we did not reproduce (no complete official code)", "",
           "| model | paper Acc | paper F1 |", "|:---|---:|---:|"]
    for name, ref in PAPER_ONLY.items():
        md.append(f"| {name} | {ref['acc']:.4f} | {ref['f1']:.4f} |")
    md += ["",
           "> Split protocol: LOSO with an 8:2:1 subject split (8 train / 2 val / 1 test "
           "subject, all 3 sessions each). Recipe: the paper's own (Adam 1e-4, batch 128, "
           "max 500 epochs) plus early stopping (patience 100), which the paper does not use. "
           "Preprocessing entry differs from the paper (official pre-segmented derivatives "
           "`.mat` vs their own 1000->250 Hz + 0.5-40 Hz). Single seed (0).", ""]
    (run_dir / "REPORT_TABLE.md").write_text("\n".join(md), encoding="utf-8")

    print(f"per_cell.csv / per_model.csv / vs_paper.csv -> {run_dir/'tables'}")
    print(f"REPORT_TABLE.md -> {run_dir/'REPORT_TABLE.md'}")
    for d in per_model:
        print(f"  {d['model']:22s} Acc {d['acc_mean']:.4f}±{d['acc_std']:.4f}  "
              f"F1 {d['f1_mean']:.4f}  ({d['n_folds']}/11 folds)")


if __name__ == "__main__":
    main()
