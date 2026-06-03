#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练 baseline 模型 EEGNet / DeepConvNet / FBCNet（Stage 2，骨架）。

先做单 session 内 train/val/test sanity check，再扩展到 subject-wise split。
GPU 作业，必须经 Slurm 提交（scripts/slurm/train_baseline_gpu.sbatch）。

用法：
  python scripts/train_baseline.py --config configs/eegnet_baseline.yaml
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
from src.utils.seed import set_seed  # noqa: E402

logger = get_logger("train_baseline")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train an EEG baseline model.")
    ap.add_argument("--config", default="configs/eegnet_baseline.yaml")
    args = ap.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    set_seed(cfg.get("train", {}).get("seed", 42))
    logger.info("run_id=%s | model=%s | mode=%s",
                cfg.get("run_id"), cfg.get("model", {}).get("name"),
                cfg.get("data", {}).get("mode"))

    # 启动即检查 CUDA（GPU 作业若 cuda 不可用应当 fail fast，见 ENVIRONMENT.md）。
    try:
        import torch
        logger.info("torch=%s cuda_avail=%s cuda_build=%s",
                    torch.__version__, torch.cuda.is_available(), torch.version.cuda)
    except ImportError:
        logger.warning("torch 未安装。")

    # TODO(Stage 2): 构建 dataset/dataloader -> 模型 -> 训练循环 -> 评估 ->
    # 保存 outputs/<run_id>/{config,metrics} 与 checkpoints/<run_id>/。
    raise NotImplementedError("train_baseline 待实现（Stage 2，见 docs/MODEL_PLAN.md）。")


if __name__ == "__main__":
    main()
