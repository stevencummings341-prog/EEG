#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""遍历 manifest 中的所有 session 做预处理（manifest 驱动，骨架/暂缓）。

读取 manifests/shu_2c_raw_manifest.csv，对每个 session 调用单 session 预处理，
保存到 configs/paths.yaml 指定的 processed 目录，并汇总写
manifests/shu_2c_processed_manifest.csv（每个 session 的形状/标签分布/状态）。
失败不静默跳过：记录 status=failed + 原因。

注意：
  - 这是重任务，必须用 Slurm CPU 作业提交（scripts/slurm/preprocess_cpu.sbatch），
    不要在登录节点直接跑全量。
  - 当前阶段（data inspector / 单 session 验证）暂不运行全量 51x3。

用法（通常由 sbatch 调用）：
  python scripts/preprocess_all.py --paths configs/paths.yaml --config configs/preprocess.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import paths  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.data.manifest import read_manifest  # noqa: E402

logger = get_logger("preprocess_all")


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess all sessions from the manifest.")
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--config", default="configs/preprocess.yaml")
    args = ap.parse_args()

    P = paths.load_paths(PROJECT_ROOT / args.paths, require_raw=True)
    _cfg = load_config(PROJECT_ROOT / args.config)  # noqa: F841 - 实现时使用
    rows = read_manifest(P.raw_manifest)
    logger.info("manifest %s -> %d sessions 待处理。", P.raw_manifest, len(rows))

    # TODO: for row in rows: 调用 preprocess_one_session，收集 (status, shape, label_counts)，
    # 写 P.processed_manifest（CSV）。失败记录原因，不静默跳过。当前阶段暂不实现/运行全量。
    raise NotImplementedError(
        "preprocess_all 暂缓（当前阶段只做单 session 验证；全量需经 Slurm CPU 作业运行）。"
    )


if __name__ == "__main__":
    main()
