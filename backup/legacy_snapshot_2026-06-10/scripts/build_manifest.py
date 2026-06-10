#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""扫描外部原始数据根，生成 manifests/shu_2c_raw_manifest.csv。

manifest 是外部 raw data 与本项目之间的桥梁。原始数据路径来自 configs/paths.yaml
（或环境变量 SHU_2C_ROOT），绝不在此写死。登录节点可运行（只做目录枚举）。

用法（项目根目录）：
  python scripts/build_manifest.py --config configs/paths.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import load_paths  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.data.manifest import build_raw_manifest, write_manifest_csv, RAW_MANIFEST_FIELDS  # noqa: E402

logger = get_logger("build_manifest")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the raw-data manifest from the external root.")
    ap.add_argument("--config", default="configs/paths.yaml", help="路径配置文件。")
    args = ap.parse_args()

    # require_raw=True：原始数据根必须存在，否则给出清晰错误提示用户填配置。
    paths = load_paths(PROJECT_ROOT / args.config, require_raw=True)
    logger.info("raw root: %s", paths.raw_root)

    rows = build_raw_manifest(paths)
    if not rows:
        raise RuntimeError(
            f"在 {paths.raw_root / paths.raw_subdir} 下未发现任何 sub-*/ses-* 。"
            " 请检查 configs/paths.yaml 的 raw_data 配置。"
        )

    n_subj = len({r["subject_id"] for r in rows})
    n_missing = sum(0 if (r["data_bdf_exists"] and r["evt_bdf_exists"]) else 1 for r in rows)
    write_manifest_csv(rows, paths.raw_manifest, fieldnames=RAW_MANIFEST_FIELDS)

    logger.info("subjects=%d, sessions=%d, missing-bdf rows=%d", n_subj, len(rows), n_missing)
    logger.info("manifest 已写入: %s", paths.raw_manifest)
    if n_missing:
        logger.warning("有 %d 个 session 缺少 data.bdf/evt.bdf，详见 manifest。", n_missing)


if __name__ == "__main__":
    main()
