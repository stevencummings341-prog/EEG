"""IO 工具：处理后数据（正式为 .npz；debug 可选 X.npy/y.npy）与 JSON 报告的读写。

保存约定：X 为 [trials, channels, time] float32（µV），y 为 [trials] int64（0/1）。
正式输出为每 session 一个 .npz，内含 X, y, subject_id, session_id, sfreq, channel_names。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    """读取一个 session 的 X/y/meta（旧 X.npy/y.npy 格式，debug/legacy 用）。"""
    out_dir = Path(out_dir)
    X = np.load(out_dir / "X.npy")
    y = np.load(out_dir / "y.npy")
    meta = load_json(out_dir / "meta.json")
    return X, y, meta


def save_session_npz(
    npz_path: str | Path,
    X: np.ndarray,
    y: np.ndarray,
    *,
    subject_id: str,
    session_id: str,
    sfreq: int,
    channel_names: List[str],
    compress: bool = True,
) -> Path:
    """把一个 session 存成正式 .npz。

    npz 内含 key：X [trials,ch,time] float32(µV)、y [trials] int64(0/1)、
    subject_id、session_id、sfreq(int)、channel_names(数组)。
    会做基本约定校验（维度/对齐）；compress=True 用 savez_compressed 省空间。
    """
    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if X.ndim != 3:
        raise ValueError(f"X must be 3D [trials, channels, time], got {X.shape}")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(f"y must be [trials] matching X[0]; got {y.shape} vs {X.shape}")
    if len(channel_names) != X.shape[1]:
        raise ValueError(
            f"channel_names ({len(channel_names)}) 必须与 X 通道数 ({X.shape[1]}) 一致。"
        )

    payload = dict(
        X=X,
        y=y,
        subject_id=np.asarray(str(subject_id)),
        session_id=np.asarray(str(session_id)),
        sfreq=np.asarray(int(sfreq), dtype=np.int64),
        channel_names=np.asarray(list(channel_names), dtype="<U16"),
    )
    saver = np.savez_compressed if compress else np.savez
    saver(npz_path, **payload)
    # np.savez* 会在没有 .npz 后缀时自动追加；规范化返回真实路径。
    if npz_path.suffix != ".npz":
        npz_path = npz_path.with_name(npz_path.name + ".npz")
    return npz_path


def load_session_npz(npz_path: str | Path) -> Dict[str, Any]:
    """读取正式 .npz，返回 dict：X, y, subject_id, session_id, sfreq, channel_names。"""
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as d:
        return {
            "X": d["X"],
            "y": d["y"],
            "subject_id": str(d["subject_id"]),
            "session_id": str(d["session_id"]),
            "sfreq": int(d["sfreq"]),
            "channel_names": [str(c) for c in d["channel_names"].tolist()],
        }


def save_debug_npy(out_dir: str | Path, X: np.ndarray, y: np.ndarray) -> None:
    """debug 可选：把 X/y 另存为 .npy（默认关闭，由 config.output.save_debug_npy 控制）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", np.asarray(X, dtype=np.float32))
    np.save(out_dir / "y.npy", np.asarray(y, dtype=np.int64))
