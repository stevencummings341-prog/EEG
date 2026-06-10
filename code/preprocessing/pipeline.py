"""单 session 预处理 + 落盘 + manifest 行组装（供 preprocess_raw / preprocess_all 复用）。

正式模式（eog_ecg_clean）每 session 落盘：
  - <sub>_<ses>_task-motorimagery_eeg.npz（X,y,subject_id,session_id,sfreq,channel_names）
  - meta.json、preprocess_report.json、（可选）manifest_row.json、（debug 可选）X.npy/y.npy

本模块函数自身不吞「形状不符」等致命异常（strict 下会抛出）；ICA/检测失败已在
eog_ecg_clean 内部退化处理。编排脚本（preprocess_all）负责 try/except 收集失败行，
保证单 session 失败不影响全量。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..data.manifest import build_processed_manifest_row
from ..utils import paths as paths_mod
from ..utils.io import save_debug_npy, save_json, save_session_npz
from .eog_ecg_clean import evaluate_failure_reasons, preprocess_one_session_eog_ecg_clean


def crosscheck_mat(P: "paths_mod.Paths", subj: str, sess: str,
                   X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """与论文 .mat 对照标签序列与形状（仅真值核对，非数据入口）。"""
    import scipy.io as sio

    mat_path = P.derivatives_mat_path(subj, sess)
    if not mat_path.exists():
        return {"checked": False, "reason": f"mat not found: {mat_path}"}
    m = sio.loadmat(str(mat_path))
    md = m["data"].transpose(2, 0, 1)
    ml = np.where(np.ravel(m["labels"]).astype(int) == 1, 0, 1)
    same_len = len(y) == len(ml)
    out: Dict[str, Any] = {
        "checked": True,
        "shape_match": bool(tuple(X.shape) == tuple(md.shape)),
        # labels_match = EXACT (same order). 注意：论文 .mat 的试次顺序未必是采集顺序，
        # 故部分 session 可能 exact=False 但 multiset=True（仅排序不同、标签内容一致）。
        # 我们以 evt.bdf 的时间顺序为准，(X_i, y_i) 同源于同一 trigger，自洽且正确。
        "labels_match": bool(same_len and np.array_equal(y, ml)),
        "labels_multiset_match": bool(same_len and np.array_equal(np.sort(y), np.sort(ml))),
        "n_labels_agree": int(np.sum(np.asarray(y) == ml)) if same_len else 0,
        "n_labels": int(len(ml)),
    }
    if out["shape_match"]:
        out["std_ours"] = float(X.std())
        out["std_paper"] = float(md.std())
    return out


def process_session_eog_ecg_clean(
    P: "paths_mod.Paths",
    cfg: Dict[str, Any],
    subj: str,
    sess: str,
    mode: str = "eog_ecg_clean",
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """处理并落盘单 session（eog_ecg_clean），返回 processed manifest 行。

    调用方需先对输出根做一次 assert_safe_output_dir + ensure_writable_dir。
    """
    out_cfg = cfg.get("output", {}) or {}
    target_sfreq = int(cfg["resample"]["target_sfreq"])
    data_bdf, evt_bdf = P.raw_bdf_paths(subj, sess)
    if not data_bdf.exists():
        raise FileNotFoundError(f"data.bdf 不存在: {data_bdf}")

    ep_cfg = cfg.get("epoch", {}) or {}
    expected_trials = int(ep_cfg.get("expected_trials", 200))
    expected_per_class = int(ep_cfg.get("expected_per_class", 100))

    X, y, meta, report = preprocess_one_session_eog_ecg_clean(
        data_bdf, evt_bdf, cfg, subject=subj, session=sess)

    if cfg.get("validate_against_mat", False):
        chk = crosscheck_mat(P, subj, sess, X, y)
        report["mat_crosscheck"] = chk
        if chk.get("checked"):
            report["labels_match_mat"] = chk.get("labels_match")
            report["labels_multiset_match"] = chk.get("labels_multiset_match")
            report["n_labels_agree"] = chk.get("n_labels_agree")
        meta["mat_crosscheck"] = chk

    # 取到 .mat 后做最终判定：multiset=False 也算失败；labels_match_mat=False 不算。
    fail_reasons = evaluate_failure_reasons(
        report["quality_checks"], int(meta.get("n_events", 0)),
        mat_crosscheck=report.get("mat_crosscheck"),
        expected_trials=expected_trials, expected_per_class=expected_per_class)
    status = "ok" if not fail_reasons else "failed"
    report["fail_reasons"] = fail_reasons
    report["status"] = status
    meta["status"] = status
    if fail_reasons:
        meta["fail_reasons"] = fail_reasons

    out_dir = P.processed_session_dir(mode, subj, sess)
    paths_mod.ensure_writable_dir(out_dir)
    npz_path = save_session_npz(
        P.session_npz_path(mode, subj, sess), X, y,
        subject_id=subj, session_id=sess, sfreq=target_sfreq,
        channel_names=meta["final_ch_names"], compress=bool(out_cfg.get("npz_compress", True)))
    meta_path = out_dir / "meta.json"
    report_path = out_dir / "preprocess_report.json"
    save_json(meta, meta_path)
    save_json(report, report_path)
    if out_cfg.get("save_debug_npy", False):
        save_debug_npy(out_dir, X, y)

    row = build_processed_manifest_row(
        subject_id=subj, session_id=sess, npz_path=npz_path,
        meta_path=meta_path, report_path=report_path, meta=meta, report=report,
        error_message="; ".join(fail_reasons))
    if out_cfg.get("write_manifest_row", True):
        save_json(row, out_dir / "manifest_row.json")

    if logger is not None:
        aux = report["aux_cleaning"]
        logger.info(
            "%s/%s | shape=%s labels=%s aux_used=%s excluded=%s match_mat=%s multiset=%s status=%s%s",
            subj, sess, meta.get("output_shape"), meta.get("label_counts"),
            aux["aux_cleaning_used"], aux["ica_excluded_components"],
            report.get("labels_match_mat"), report.get("labels_multiset_match"),
            row["status"], (" reasons=" + "; ".join(fail_reasons)) if fail_reasons else "")
    return row
