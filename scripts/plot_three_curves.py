#!/usr/bin/env python
"""Plot per-epoch train / validation / test accuracy for a cross-subject run.

The advisor's request (2026-08-09): "train 和 validate，同时测试一下测试集的准确率,
然后我们可以同时可视化三者的准确率差异".

The three curves come from ``history`` in each cell's ``result.json``, written by
``code/training/e2e_trainer.py`` when ``train.curves: true``:

  * ``train_eval`` — capped training subset, eval mode (comparable to the other two)
  * ``val``        — validation split; the ONLY curve model selection ever saw
  * ``test``       — held-out subject, **monitoring only**, never used for selection

Outputs, under ``<run_dir>/figures/``:
  * ``curves__<model>__fold<k>.png``  — one panel per requested cell
  * ``curves__<model>__mean.png``     — mean over folds (per-epoch, folds truncated to the
                                        shortest one so the mean is over a constant set)

Usage:
  python scripts/plot_three_curves.py --run outputs/experiments/wbci_shu/paper_baseline_3c_821_v1
  python scripts/plot_three_curves.py --run <dir> --models atcnet,eegnex --folds 0,6,10
  python scripts/plot_three_curves.py --run <dir> --metric macro_f1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SPLITS = [("train_eval", "train", "tab:blue"),
          ("val", "validation", "tab:orange"),
          ("test", "test (monitor only)", "tab:green")]


def _parse_list(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def load_cell(result_json: Path) -> Dict:
    with open(result_json, "r", encoding="utf-8") as f:
        return json.load(f)


def series(history: List[Dict], split: str, metric: str) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for row in history:
        block = row.get(split) or {}
        v = block.get(metric)
        out.append(float(v) if v is not None else None)
    return out


def plot_cell(history: List[Dict], metric: str, title: str, out_png: Path,
              best_epoch: Optional[int] = None) -> bool:
    epochs = [int(r["epoch"]) for r in history]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    plotted = False
    for key, label, color in SPLITS:
        ys = series(history, key, metric)
        if all(y is None for y in ys):
            continue
        ax.plot(epochs, ys, label=label, color=color, linewidth=1.4)
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    if best_epoch and best_epoch > 0:
        ax.axvline(best_epoch, color="grey", linestyle="--", linewidth=1.0,
                   label=f"best.pt (epoch {best_epoch})")
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    return True


def plot_model_mean(histories: List[List[Dict]], metric: str, title: str,
                    out_png: Path) -> bool:
    if not histories:
        return False
    # Truncate to the shortest history: with early stopping the folds differ in length, and
    # averaging over a shrinking set of folds would bend the tail of the curve.
    n = min(len(h) for h in histories)
    if n < 2:
        return False
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    plotted = False
    for key, label, color in SPLITS:
        cols = []
        for h in histories:
            ys = series(h[:n], key, metric)
            if any(y is None for y in ys):
                cols = []
                break
            cols.append(ys)
        if not cols:
            continue
        mean = [sum(c[i] for c in cols) / len(cols) for i in range(n)]
        ax.plot(range(1, n + 1), mean, label=f"{label} (mean of {len(cols)} folds)",
                color=color, linewidth=1.6)
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(f"{title}  [epochs truncated to shortest fold = {n}]", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run dir (contains cells/ and figures/)")
    ap.add_argument("--models", help="comma list; default = all models found")
    ap.add_argument("--folds", help="comma list of fold ids; default = all")
    ap.add_argument("--metric", default="accuracy", help="metric to plot (default: accuracy)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run)
    cells_dir = run_dir / "cells"
    if not cells_dir.is_dir():
        raise SystemExit(f"no cells/ under {run_dir}")

    want_models = _parse_list(args.models)
    want_folds = {int(f) for f in _parse_list(args.folds)} if args.folds else None
    fig_dir = run_dir / "figures"

    per_model: Dict[str, List[List[Dict]]] = {}
    n_cell_png = 0
    skipped_no_curves = 0

    for result_json in sorted(cells_dir.glob(f"*__seed{args.seed}/result.json")):
        tag = result_json.parent.name                       # <model>__fold<k>__seed<s>
        model, fold_part, _ = tag.split("__")
        fold = int(fold_part.replace("fold", ""))
        if want_models and model not in want_models:
            continue
        if want_folds is not None and fold not in want_folds:
            continue

        detail = load_cell(result_json)
        history = detail.get("history") or []
        if not history or "test" not in (history[0] or {}):
            skipped_no_curves += 1
            continue

        test_subj = ",".join((detail.get("fold") or {}).get("test_subjects") or [])
        best_epoch = (detail.get("train_info") or {}).get("best_epoch")
        title = f"{model}  fold{fold} (test={test_subj})  —  {args.metric}"
        if plot_cell(history, args.metric, title,
                     fig_dir / f"curves__{model}__fold{fold}.png", best_epoch):
            n_cell_png += 1
        per_model.setdefault(model, []).append(history)

    n_mean_png = 0
    for model, histories in sorted(per_model.items()):
        if plot_model_mean(histories, args.metric,
                           f"{model} — {args.metric}, mean over folds",
                           fig_dir / f"curves__{model}__mean.png"):
            n_mean_png += 1

    print(f"wrote {n_cell_png} per-cell + {n_mean_png} per-model figures -> {fig_dir}")
    if skipped_no_curves:
        print(f"skipped {skipped_no_curves} cells without curve data "
              "(trained before train.curves was enabled)")


if __name__ == "__main__":
    main()
