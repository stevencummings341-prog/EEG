#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 41/10 subject-wise split JSON（多 seed），持久化到 configs/paths.yaml 的 splits 目录。

数据入口 = processed_manifest.csv 里 status == ok 的 per-session .npz（derivatives .mat 仅 QC）。
铁律：按被试划分；target 被试 3 session 全 ok；source 允许含 failed session 但只用 ok session。

用法（登录节点即可，只读 CSV、写 JSON，秒级）：
  python scripts/make_splits.py --paths configs/paths.yaml \
      --seeds 2026 2027 2028 2029 2030 --run-id cap_eegnet_4110
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import make_and_save_splits, load_split  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.utils.paths import load_paths  # noqa: E402

logger = get_logger("make_splits")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate 41/10 subject-wise split JSONs.")
    ap.add_argument("--paths", default="configs/paths.yaml", help="路径配置 YAML。")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029, 2030])
    ap.add_argument("--n-source", type=int, default=41)
    ap.add_argument("--n-target", type=int, default=10)
    ap.add_argument("--run-id", default="cap_eegnet_4110")
    ap.add_argument("--variant", default="eog_ecg_clean")
    args = ap.parse_args()

    # 只需要 processed_manifest + splits 目录，不必校验 raw 根存在。
    P = load_paths(PROJECT_ROOT / args.paths, require_raw=False)
    logger.info("manifest = %s", P.processed_manifest)
    logger.info("splits dir = %s", P.splits_dir)

    out_paths = make_and_save_splits(
        P.processed_manifest,
        P.splits_dir,
        seeds=args.seeds,
        n_source=args.n_source,
        n_target=args.n_target,
        run_id=args.run_id,
        variant=args.variant,
    )

    logger.info("已生成 %d 个 split JSON：", len(out_paths))
    for p in out_paths:
        s = load_split(p)
        logger.info(
            "  %s | seed=%s | source=%d target=%d | excluded=%d | target=%s",
            p.name, s["seed"], s["n_source_subjects"], s["n_target_subjects"],
            s["counts"]["excluded_sessions"], s["target_subjects"],
        )


if __name__ == "__main__":
    main()
