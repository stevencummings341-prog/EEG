#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""遍历 51x3 所有 session 做预处理（Task 3，骨架）。

遍历 data/raw 对应的数据集 sourcedata/2C dataset/sub-*/ses-*/eeg/，对每个 session
调用单 session 预处理，保存到 data/processed_paper_style，并汇总到
outputs/preprocess_summary.csv（每个 session 的形状/标签分布/状态）。

注意：这是重任务，必须用 Slurm CPU 作业提交（scripts/slurm/preprocess_cpu.sbatch），
不要在登录节点直接跑全量。

用法（通常由 sbatch 调用）：
  python scripts/preprocess_all.py --config configs/preprocess.yaml
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

logger = get_logger("preprocess_all")


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess all subjects/sessions.")
    ap.add_argument("--config", default="configs/preprocess.yaml")
    args = ap.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)  # noqa: F841 - 实现时使用
    sessions = list(paths.iter_sessions())
    logger.info("发现 %d 个被试，共 %d 个 session 待处理。",
                len(paths.list_subjects()), len(sessions))

    # TODO(Task 3): for (subj, sess) in sessions: 调用单 session 预处理，收集结果，
    # 失败不静默跳过（记录 status=failed + 原因），最后写 outputs/preprocess_summary.csv。
    raise NotImplementedError("preprocess_all 待实现（Task 3，依赖 Task 2 的单 session 逻辑）。")


if __name__ == "__main__":
    main()
