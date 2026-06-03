#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""41/10 跨被试预训练（Experiment A / Stage 3，骨架）。

随机选 41 source / 10 target（按被试划分，持久化到 outputs/splits/），
在 source 全部 session 上训练，在 target 全部 session 上 zero-shot 测试；
多 seed 重复，报告 mean ± std。GPU 作业，经 Slurm 提交。

用法：
  python scripts/train_cross_subject.py --config configs/cross_subject.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger("train_cross_subject")


def main() -> None:
    ap = argparse.ArgumentParser(description="41/10 cross-subject pretraining.")
    ap.add_argument("--config", default="configs/cross_subject.yaml")
    args = ap.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    logger.info("run_id=%s | seeds=%s | source=%d target=%d",
                cfg.get("run_id"), cfg.get("split", {}).get("seeds"),
                cfg.get("split", {}).get("n_source_subjects"),
                cfg.get("split", {}).get("n_target_subjects"))

    # TODO(Stage 3): 对每个 seed 生成 subject-wise split（src.data.shu_dataset），
    # 训练 -> zero-shot 评估 target -> 汇总 mean±std。严禁 target 泄漏进 source。
    raise NotImplementedError("train_cross_subject 待实现（Stage 3，见 docs/EXPERIMENT_PROTOCOL.md）。")


if __name__ == "__main__":
    main()
