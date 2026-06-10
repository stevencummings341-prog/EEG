"""配置加载工具：读取 configs/*.yaml。

约定（90-agent-behavior 规则）：脚本通过 argparse 接收 config 路径，
关键超参数全部来自 YAML，不在脚本里写死。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """读取并解析一个 YAML 配置文件，返回 dict。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config must parse to a dict, got {type(cfg)}: {path}")
    return cfg


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    """把（解析/合并后的）配置原样写出，便于每次 run 留档。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
