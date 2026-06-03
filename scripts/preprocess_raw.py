#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对单个 subject/session 做 Python 预处理。

输入一个 session 的 data.bdf + evt.bdf（路径来自 configs/paths.yaml），输出
X.npy / y.npy / meta.json 到 configs/paths.yaml 指定的 processed 目录，
要求 X.shape == [200, 58, 1000]；得不到该形状必须报错并记录原因。

可选：用 derivatives 里的论文 .mat 做标签对照（仅真值核对，非数据入口）。

这是当前阶段的目标脚本：用 Python 从 raw 跑出 [200,58,1000]。
注意：单 session 也比 30s 重，应在计算节点用 srun 运行，不要在登录节点跑。

用法：
  python scripts/preprocess_raw.py --paths configs/paths.yaml \
      --config configs/preprocess.yaml --subject 1 --session 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import paths  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.io import save_processed_session, save_json  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.preprocessing.shu_preprocess import preprocess_one_session  # noqa: E402

logger = get_logger("preprocess_raw")


def _validate_against_mat(P: "paths.Paths", subj: str, sess: str,
                          X: np.ndarray, y: np.ndarray) -> dict:
    """与论文 .mat 对照标签序列与形状（仅真值核对）。"""
    import scipy.io as sio

    mat_path = P.derivatives_mat_path(subj, sess)
    if not mat_path.exists():
        return {"checked": False, "reason": f"mat not found: {mat_path}"}
    m = sio.loadmat(str(mat_path))
    md = m["data"].transpose(2, 0, 1)                       # [trials, ch, time]
    ml = np.where(np.ravel(m["labels"]).astype(int) == 1, 0, 1)
    out = {
        "checked": True,
        "shape_match": bool(tuple(X.shape) == tuple(md.shape)),
        "labels_match": bool(len(y) == len(ml) and np.array_equal(y, ml)),
    }
    if out["shape_match"]:
        out["std_ours"] = float(X.std())
        out["std_paper"] = float(md.std())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess one subject/session (paper-style, Python).")
    ap.add_argument("--paths", default="configs/paths.yaml", help="路径配置。")
    ap.add_argument("--config", default="configs/preprocess.yaml", help="预处理参数配置。")
    ap.add_argument("--subject", required=True, help="如 1 或 sub-001")
    ap.add_argument("--session", required=True, help="如 1 或 ses-01")
    args = ap.parse_args()

    P = paths.load_paths(PROJECT_ROOT / args.paths, require_raw=True)
    cfg = load_config(PROJECT_ROOT / args.config)
    variant = cfg.get("variant", "paper_style")
    subj, sess = paths.sub_id(args.subject), paths.ses_id(args.session)
    data_bdf, evt_bdf = P.raw_bdf_paths(subj, sess)
    logger.info("paths=%s config=%s | %s/%s | variant=%s", args.paths, args.config, subj, sess, variant)
    logger.info("data.bdf=%s", data_bdf)
    if not data_bdf.exists():
        raise FileNotFoundError(f"data.bdf 不存在: {data_bdf}（检查 configs/paths.yaml）")

    X, y, meta = preprocess_one_session(data_bdf, evt_bdf, cfg, subject=subj, session=sess)
    logger.info("X.shape=%s y.shape=%s label_counts=%s status=%s",
                X.shape, y.shape, meta.get("label_counts"), meta.get("status"))

    # 可选：与论文 .mat 标签对照。
    if cfg.get("validate_against_mat", False):
        chk = _validate_against_mat(P, subj, sess, X, y)
        meta["mat_crosscheck"] = chk
        logger.info("mat 对照: %s", chk)

    # 输出到 configs/paths.yaml 指定的 processed 目录。
    out_dir = P.processed_session_dir(variant, subj, sess)
    save_processed_session(out_dir, X, y, meta)
    logger.info("已保存到 %s", out_dir)

    # 追加一行 processed manifest（单 session 也写，便于增量累积）。
    row = {
        "subject_id": subj, "session_id": sess, "variant": variant,
        "x_path": str(out_dir / "X.npy"), "y_path": str(out_dir / "y.npy"),
        "n_trials": int(X.shape[0]), "n_channels": int(X.shape[1]),
        "n_times": int(X.shape[2]), "status": meta.get("status"),
    }
    save_json(row, out_dir / "manifest_row.json")
    logger.info("processed manifest 行: %s", row)


if __name__ == "__main__":
    main()
