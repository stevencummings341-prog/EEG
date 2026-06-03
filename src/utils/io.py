"""IO 工具：处理后数据 (X.npy/y.npy/meta.json) 与 JSON 报告的读写。

保存约定：X 为 [trials, channels, time] float32，y 为 [trials] int64。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    """写 JSON（utf-8，缩进，支持中文，numpy 标量自动转原生类型）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any):
    """让 json 能序列化 numpy 类型与 Path。"""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def save_processed_session(
    out_dir: str | Path,
    X: np.ndarray,
    y: np.ndarray,
    meta: Dict[str, Any],
) -> None:
    """保存一个 session 的处理结果到 out_dir/{X.npy, y.npy, meta.json}。

    会校验 X/y 的基本约定（dtype 与 trial 数一致），不满足则报错。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if X.ndim != 3:
        raise ValueError(f"X must be 3D [trials, channels, time], got {X.shape}")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(f"y must be [trials] matching X[0]; got {y.shape} vs {X.shape}")

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    save_json(meta, out_dir / "meta.json")


def load_processed_session(out_dir: str | Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """读取一个 session 的 X/y/meta。"""
    out_dir = Path(out_dir)
    X = np.load(out_dir / "X.npy")
    y = np.load(out_dir / "y.npy")
    meta = load_json(out_dir / "meta.json")
    return X, y, meta
