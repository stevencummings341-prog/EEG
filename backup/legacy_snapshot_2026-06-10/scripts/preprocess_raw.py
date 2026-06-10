#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对单个 subject/session 做 Python 预处理（正式：mode=eog_ecg_clean -> .npz）。

输入一个 session 的 data.bdf + evt.bdf（路径来自 configs/paths.yaml）。正式模式
eog_ecg_clean 输出每 session 一个 .npz（X,y,subject_id,session_id,sfreq,channel_names）
+ meta.json + preprocess_report.json（+ 可选 manifest_row.json），要求
X.shape == [200, 58, 1000]；得不到该形状报错并记录原因。

输出只写入 configs/paths.yaml 指定的 processed 目录；脚本会拒绝写入 raw 的
sourcedata/derivatives，目录不可写则直接报错。

注意：单 session 也比 30s 重（尤其 ICA），必须在计算节点用 srun/sbatch 运行。

用法：
  python scripts/preprocess_raw.py --paths configs/paths.yaml \
      --config configs/preprocess.yaml --subject 1 --session 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import paths  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.io import save_processed_session, save_json  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.preprocessing.pipeline import crosscheck_mat, process_session_eog_ecg_clean  # noqa: E402
from src.preprocessing.shu_preprocess import preprocess_one_session  # noqa: E402

logger = get_logger("preprocess_raw")


def run_paper_style(P, cfg, subj, sess, data_bdf, evt_bdf, mode) -> None:
    """旧的 paper-style（无 ICA）：保留 X.npy/y.npy 输出，作 sanity check 用。"""
    out_dir = P.processed_session_dir(mode, subj, sess)
    paths.assert_safe_output_dir(out_dir, P)
    X, y, meta = preprocess_one_session(data_bdf, evt_bdf, cfg, subject=subj, session=sess)
    logger.info("X.shape=%s y.shape=%s label_counts=%s status=%s",
                X.shape, y.shape, meta.get("label_counts"), meta.get("status"))
    if cfg.get("validate_against_mat", False):
        meta["mat_crosscheck"] = crosscheck_mat(P, subj, sess, X, y)
        logger.info("mat 对照: %s", meta["mat_crosscheck"])
    save_processed_session(out_dir, X, y, meta)
    save_json({
        "subject_id": subj, "session_id": sess, "variant": mode,
        "x_path": str(out_dir / "X.npy"), "y_path": str(out_dir / "y.npy"),
        "n_trials": int(X.shape[0]), "n_channels": int(X.shape[1]),
        "n_times": int(X.shape[2]), "status": meta.get("status"),
    }, out_dir / "manifest_row.json")
    logger.info("已保存到 %s", out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess one subject/session (Python).")
    ap.add_argument("--paths", default="configs/paths.yaml", help="路径配置。")
    ap.add_argument("--config", default="configs/preprocess.yaml", help="预处理参数配置。")
    ap.add_argument("--subject", required=True, help="如 1 或 sub-001")
    ap.add_argument("--session", required=True, help="如 1 或 ses-01")
    args = ap.parse_args()

    P = paths.load_paths(PROJECT_ROOT / args.paths, require_raw=True)
    cfg = load_config(PROJECT_ROOT / args.config)
    mode = cfg.get("mode", cfg.get("variant", "paper_style"))
    subj, sess = paths.sub_id(args.subject), paths.ses_id(args.session)
    data_bdf, evt_bdf = P.raw_bdf_paths(subj, sess)
    logger.info("paths=%s config=%s | %s/%s | mode=%s", args.paths, args.config, subj, sess, mode)
    logger.info("data.bdf=%s", data_bdf)
    if not data_bdf.exists():
        raise FileNotFoundError(f"data.bdf 不存在: {data_bdf}（检查 configs/paths.yaml）")

    if mode == "eog_ecg_clean":
        out_root = P.processed_dir(mode)
        paths.assert_safe_output_dir(out_root, P)
        paths.ensure_writable_dir(out_root)
        row = process_session_eog_ecg_clean(P, cfg, subj, sess, mode=mode, logger=logger)
        logger.info("manifest 行: %s", row)
        if row.get("status") != "ok":
            logger.error("session 判失败: %s", row.get("error_message"))
            sys.exit(1)
    else:
        run_paper_style(P, cfg, subj, sess, data_bdf, evt_bdf, mode)


if __name__ == "__main__":
    main()
