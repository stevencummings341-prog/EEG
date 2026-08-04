#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified experiment runner for the modular EEG-MI framework.

唯一人工入口。`--dry-run` 解析配置并打印计划；正常运行时在进程内直接调用
`code/runners.py` 的 phase runner（不依赖已归档的旧 `scripts/`）。

输入: code/configs/experiments/*.yaml
输出: 各 phase config 指定的 outputs/checkpoints 目录。
依赖: pyyaml, torch/numpy/scipy/sklearn/pandas（具体实验需要）。

示例:
  python code/run.py --dry-run --config code/configs/experiments/phase1_baseline.yaml
  python code/run.py --config code/configs/experiments/phase0_drift_diagnostic.yaml --subjects 1,2
  python code/run.py --config code/configs/experiments/phase1_baseline.yaml \
      --models eegnet --subjects 1,2 --folds 2 --max-epochs 3 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import yaml

CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PHASES = [
    "phase0_drift_diagnostic",
    "phase1_baseline",
    "phase2a_multisource",
    "phase2b_alignment",
]


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def experiment_name(cfg: Dict, path: Path) -> str:
    exp = cfg.get("experiment") or {}
    if exp.get("name"):
        return str(exp["name"])
    return path.stem


def summarize_config(cfg: Dict, cfg_path: Path) -> Dict[str, object]:
    from code.runners import PHASE_RUNNERS

    exp = experiment_name(cfg, cfg_path)
    data = cfg.get("data") or cfg.get("dataset") or {}
    models = cfg.get("models")
    if models is None and cfg.get("model"):
        models = [cfg["model"].get("name")]
    return {
        "experiment": exp,
        "config": str(cfg_path),
        "dataset": data.get("name") or data.get("variant") or cfg.get("dataset", {}).get("name"),
        "models": models,
        "output": cfg.get("output", {}),
        "runnable_now": exp in PHASE_RUNNERS,
        "runner": f"code.runners.{exp}" if exp in PHASE_RUNNERS else "not yet implemented",
    }


def run_one(cfg_path: Path, args: argparse.Namespace) -> None:
    from code.runners import PHASE_RUNNERS

    cfg_path = cfg_path.resolve()
    cfg = load_yaml(cfg_path)
    exp = experiment_name(cfg, cfg_path)
    if args.dry_run:
        print(json.dumps(summarize_config(cfg, cfg_path), ensure_ascii=False, indent=2))
        return
    if args.summarize:
        from code.summaries.summarize import summarize_phase
        print(f"[summarize] {exp} <- {cfg_path}")
        summarize_phase(exp, cfg)
        return
    runner = PHASE_RUNNERS.get(exp)
    if runner is None:
        raise SystemExit(
            f"No runner registered for experiment '{exp}'. "
            f"Known: {sorted(PHASE_RUNNERS)}. Add one in code/runners.py."
        )
    print(f"[run] {exp} <- {cfg_path}")
    runner(cfg, cfg_path, args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified EEG-MI experiment runner")
    parser.add_argument("--config", help="one code/configs/experiments/*.yaml file")
    parser.add_argument("--all", action="store_true", help="run all registered phases in order")
    parser.add_argument("--dry-run", action="store_true", help="resolve config and print plan only")
    parser.add_argument("--summarize", action="store_true",
                        help="aggregate existing run outputs into tables/figures + canonical REPORT.md")
    parser.add_argument(
        "--paths",
        default="code/configs/paths.yaml",
        help="paths config (auto-prefers paths.local.yaml if present)",
    )
    parser.add_argument("--device", help="cuda | cpu | auto")
    parser.add_argument("--models", help="comma-separated model names")
    parser.add_argument("--methods", help="comma-separated alignment methods (phase2b)")
    parser.add_argument("--subjects", help="comma-separated subject ids, e.g. 1,2")
    parser.add_argument("--sessions", help="comma-separated sessions, e.g. 1,2,3")
    parser.add_argument("--seeds", help="comma-separated seeds")
    parser.add_argument("--folds", type=int, help="within-session K (phase1)")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--val-fraction", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--protocol", choices=["within", "cross", "both"], help="phase1 protocol")
    parser.add_argument("--protocols", help="phase2b protocol groups: single,multi")
    parser.add_argument("--ckpt-dir", help="override checkpoint dir (phase2b)")
    parser.add_argument("--tag-suffix", default="", help="appended to run CSV names (phase2b)")
    parser.add_argument("--out", help="override output_dir")
    parser.add_argument("--no-save-ckpt", action="store_true", help="do not write checkpoints")
    args = parser.parse_args()

    if args.all:
        for phase in DEFAULT_PHASES:
            cfg_path = CODE_ROOT / "configs" / "experiments" / f"{phase}.yaml"
            if cfg_path.exists():
                run_one(cfg_path, args)
        return
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = PROJECT_ROOT / cfg_path
        run_one(cfg_path, args)
        return
    parser.print_help()


if __name__ == "__main__":
    main()
