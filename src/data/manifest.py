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

# 正式（eog_ecg_clean）processed manifest 字段：每行一个 session，记录 npz_path（不再
# 记录 X_path/y_path）。详细分数等在 preprocess_report.json 里。
PROCESSED_MANIFEST_FIELDS = [
    "subject_id",
    "session_id",
    "npz_path",
    "meta_path",
    "report_path",
    "status",
    "n_trials",
    "n_channels",
    "n_times",
    "sfreq",
    "label_0_count",
    "label_1_count",
    "labels_match_mat",
    "labels_multiset_match",
    "n_labels_agree",
    "aux_cleaning_used",
    "valid_eog_channels",
    "valid_ecg_channels",
    "ica_excluded_components",
    "error_message",
]


def _join_list(xs: Any) -> str:
    """把列表压成 CSV 友好的 '|' 串（None/空 -> ''）。"""
    if not xs:
        return ""
    return "|".join(str(x) for x in xs)


def build_processed_manifest_row(
    *,
    subject_id: str,
    session_id: str,
    npz_path: Any = "",
    meta_path: Any = "",
    report_path: Any = "",
    meta: Dict[str, Any] | None = None,
    report: Dict[str, Any] | None = None,
    status: str | None = None,
    error_message: str = "",
) -> Dict[str, Any]:
    """从 meta/report 组装一行 processed manifest（成功/失败都能用）。"""
    meta = meta or {}
    report = report or {}
    qc = report.get("quality_checks", {}) or {}
    aux = report.get("aux_cleaning", {}) or {}
    shape = qc.get("shape") or meta.get("output_shape") or []
    return {
        "subject_id": subject_id,
        "session_id": session_id,
        "npz_path": str(npz_path) if npz_path else "",
        "meta_path": str(meta_path) if meta_path else "",
        "report_path": str(report_path) if report_path else "",
        "status": status or report.get("status") or meta.get("status") or "",
        "n_trials": meta.get("n_trials", shape[0] if shape else ""),
        "n_channels": qc.get("n_channels", shape[1] if len(shape) > 1 else ""),
        "n_times": qc.get("n_times", shape[2] if len(shape) > 2 else ""),
        "sfreq": qc.get("sfreq", meta.get("target_sfreq", "")),
        "label_0_count": qc.get("label_0_count", ""),
        "label_1_count": qc.get("label_1_count", ""),
        "labels_match_mat": report.get("labels_match_mat", ""),
        "labels_multiset_match": report.get("labels_multiset_match", ""),
        "n_labels_agree": report.get("n_labels_agree", ""),
        "aux_cleaning_used": aux.get("aux_cleaning_used", meta.get("aux_cleaning_used", "")),
        "valid_eog_channels": _join_list(aux.get("valid_eog_channels")),
        "valid_ecg_channels": _join_list(aux.get("valid_ecg_channels")),
        "ica_excluded_components": _join_list(aux.get("ica_excluded_components")),
        "error_message": error_message,
    }


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
