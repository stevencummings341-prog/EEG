"""日志工具：统一的 logger 配置（控制台 + 可选文件）。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str = "eegmi", logfile: str | Path | None = None,
               level: int = logging.INFO) -> logging.Logger:
    """返回配置好的 logger。重复调用不会重复添加 handler。"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    if logfile is not None:
        logfile = Path(logfile)
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
