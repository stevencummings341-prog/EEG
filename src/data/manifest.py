"""Manifest 工具：扫描外部原始数据，生成/读取 raw 与 processed manifest。

manifest 是「外部原始数据」与「本项目」之间的桥梁（见 10-data-paths 规则）。
raw manifest 至少包含：subject_id, session_id, data_bdf_path, evt_bdf_path。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

from ..utils.paths import Paths

RAW_MANIFEST_FIELDS = [
    "subject_id",
    "session_id",
    "data_bdf_path",
    "evt_bdf_path",
    "data_bdf_exists",
    "evt_bdf_exists",
    "data_bdf_size_bytes",
]


def build_raw_manifest(paths: Paths) -> List[Dict[str, Any]]:
    """扫描外部原始数据根，枚举所有 subject/session，返回 manifest 行。

    被试与 session 都从磁盘枚举（不写死数量）。每行记录 BDF 路径与是否存在。
    """
    rows: List[Dict[str, Any]] = []
    for subj, sess in paths.iter_sessions():
        data_bdf, evt_bdf = paths.raw_bdf_paths(subj, sess)
        rows.append({
            "subject_id": subj,
            "session_id": sess,
            "data_bdf_path": str(data_bdf),
            "evt_bdf_path": str(evt_bdf),
            "data_bdf_exists": data_bdf.exists(),
            "evt_bdf_exists": evt_bdf.exists(),
            "data_bdf_size_bytes": data_bdf.stat().st_size if data_bdf.exists() else 0,
        })
    return rows


def write_manifest_csv(rows: List[Dict[str, Any]], path: str | Path,
                       fieldnames: List[str] | None = None) -> None:
    """写 manifest CSV。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else RAW_MANIFEST_FIELDS
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: str | Path) -> List[Dict[str, str]]:
    """读 manifest CSV，返回行 dict 列表。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"manifest 不存在: {path}。先运行 scripts/build_manifest.py 生成。"
        )
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
