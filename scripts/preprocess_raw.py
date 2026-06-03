#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对单个 subject/session 做预处理（Task 2，骨架）。

输入一个 session 的 data.bdf + evt.bdf，输出 X.npy / y.npy / meta.json，
要求 X.shape == [200, 58, 1000]；得不到该形状必须报错并记录原因。

实现见 src/preprocessing/shu_preprocess.py 与 docs/PREPROCESSING_SPEC.md。
先用 scripts/check_raw_bdf.py 确认通道角色、事件数与切窗时机，再实现本脚本。

用法：
  python scripts/preprocess_raw.py --config configs/preprocess.yaml --subject 1 --session 1
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
from src.utils.io import save_processed_session  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.preprocessing.shu_preprocess import preprocess_one_session  # noqa: E402

logger = get_logger("preprocess_raw")


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess one subject/session (paper-style).")
    ap.add_argument("--config", default="configs/preprocess.yaml")
    ap.add_argument("--subject", required=True, help="如 1 或 sub-001")
    ap.add_argument("--session", required=True, help="如 1 或 ses-01")
    args = ap.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    variant = cfg.get("variant", "paper_style")
    subj, sess = paths.sub_id(args.subject), paths.ses_id(args.session)
    data_bdf, evt_bdf = paths.raw_bdf_paths(subj, sess)
    logger.info("config=%s | %s/%s | variant=%s", args.config, subj, sess, variant)
    logger.info("data.bdf=%s", data_bdf)
    if not data_bdf.exists():
        raise FileNotFoundError(f"data.bdf 不存在: {data_bdf}")

    X, y, meta = preprocess_one_session(data_bdf, evt_bdf, cfg, subject=subj, session=sess)
    logger.info("X.shape=%s y.shape=%s label_counts=%s status=%s",
                X.shape, y.shape, meta.get("label_counts"), meta.get("status"))

    # 输出目录：data/<processed_variant>/sub-XXX/ses-YY/
    variant_dir = {
        "paper_style": "processed_paper_style",
        "eog_ecg_clean": "processed_eog_ecg_clean",
    }.get(variant, f"processed_{variant}")
    out_dir = paths.project_path("data", variant_dir, subj, sess)
    save_processed_session(out_dir, X, y, meta)
    logger.info("已保存到 %s", out_dir)


if __name__ == "__main__":
    main()
