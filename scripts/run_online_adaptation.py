#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在线 test-then-update 适应（Experiment C / Stage 5，骨架）。

从预训练（可选 Session 1 微调）模型出发，在 target 的 Session 2、Session 3 上
按时间顺序逐 trial 在线学习。铁律：每个 trial 先预测+记录置信度+记录正确性，
再更新；只更新轻量模块（prototype / adapter / calibration head / BN / 有标签时分类头），
默认冻结 backbone。GPU 作业，经 Slurm 提交。

用法：
  python scripts/run_online_adaptation.py --config configs/online_adaptation.yaml
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

logger = get_logger("run_online_adaptation")


def main() -> None:
    ap = argparse.ArgumentParser(description="Online test-then-update adaptation (CAP-EEGNet).")
    ap.add_argument("--config", default="configs/online_adaptation.yaml")
    args = ap.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    online = cfg.get("online", {})
    logger.info("run_id=%s | label_mode=%s | conf_thr=%s | update=%s",
                cfg.get("run_id"), online.get("label_mode"),
                online.get("confidence_threshold"), online.get("update_modules"))

    # TODO(Stage 5): 逐 trial：predict -> record(pred,conf,correct) -> update（门控）。
    # 严禁在评估前用该 trial 训练；记录 per_trial.csv 与 pre/post 性能曲线。
    raise NotImplementedError("run_online_adaptation 待实现（Stage 5，见 docs/EXPERIMENT_PROTOCOL.md）。")


if __name__ == "__main__":
    main()
