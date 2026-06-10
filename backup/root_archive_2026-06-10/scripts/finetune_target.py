#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""目标被试微调（Experiment B / Stage 4，骨架）。

从跨被试预训练 checkpoint 出发，对每个 target subject 用 Session 1 微调，
在 Session 2 + 3 测试。比较 zero-shot / classifier-only / adapter / prototype /
full-model 等策略。GPU 作业，经 Slurm 提交。

用法：
  python scripts/finetune_target.py --config configs/finetune.yaml
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

logger = get_logger("finetune_target")


def main() -> None:
    ap = argparse.ArgumentParser(description="Target-subject fine-tuning.")
    ap.add_argument("--config", default="configs/finetune.yaml")
    args = ap.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    logger.info("run_id=%s | strategy=%s | budget=%s",
                cfg.get("run_id"), cfg.get("strategy"), cfg.get("data_budget"))
    if not cfg.get("pretrained", {}).get("checkpoint"):
        logger.warning("未指定 pretrained.checkpoint —— 微调需要预训练权重。")

    # TODO(Stage 4): 载入预训练权重 -> 按 strategy 选择可训练参数 -> 用 Session 1 微调
    # -> 在 Session 2/3 评估（含校准指标）-> 按 target subject 汇总。
    raise NotImplementedError("finetune_target 待实现（Stage 4，见 docs/EXPERIMENT_PROTOCOL.md）。")


if __name__ == "__main__":
    main()
