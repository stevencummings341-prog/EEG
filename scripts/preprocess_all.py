#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""遍历 raw manifest 做正式预处理（mode=eog_ecg_clean），每 session 输出 .npz。

读取 manifests/shu_2c_raw_manifest.csv，对每个 session 调用 EOG/ECG 辅助降噪预处理，
落盘到 configs/paths.yaml 的 eog_ecg_clean_root 下 sub-XXX/ses-YY/：
  <sub>_<ses>_task-motorimagery_eeg.npz + meta.json + preprocess_report.json (+ manifest_row.json)
并在该根目录汇总写 processed_manifest.csv 与 preprocess_summary.csv（每个 session 一行，
记录 npz_path / 形状 / 标签分布 / aux 是否启用 / ICA 排除分量 / 状态 / 错误信息）。
单 session 失败不静默跳过、也不影响全量：记录 status=failed + error_message。

注意：
  - 这是重任务（每 session 含 ICA），必须用 Slurm CPU 作业或 srun 在计算节点跑，
    不要在登录节点直接跑全量 51x3。
  - dry-run 小规模：用 --subjects / --sessions / --limit 过滤；--tag 区分输出文件名。

用法：
  # 全量（计算节点）：
  python scripts/preprocess_all.py --paths configs/paths.yaml --config configs/preprocess.yaml
  # dry-run sub-001 三个 session（计算节点，srun）：
  python scripts/preprocess_all.py --subjects 1 --tag dryrun
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import paths  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.data.manifest import (  # noqa: E402
    PROCESSED_MANIFEST_FIELDS, build_processed_manifest_row, read_manifest, write_manifest_csv,
)
from src.preprocessing.pipeline import process_session_eog_ecg_clean  # noqa: E402

logger = get_logger("preprocess_all")

SUMMARY_FIELDS = [
    "subject_id", "session_id", "status", "n_trials", "n_channels", "n_times",
    "sfreq", "label_0_count", "label_1_count", "labels_match_mat",
    "labels_multiset_match", "n_labels_agree", "aux_cleaning_used",
    "ica_excluded_components", "error_message",
]


def _parse_filter(arg: str | None, norm) -> set[str] | None:
    """把 '1,2' 或 'sub-001,sub-002' 规范化成集合；None 表示不过滤。"""
    if not arg:
        return None
    return {norm(x.strip()) for x in arg.split(",") if x.strip()}


def _tagged(path: Path, tag: str | None) -> Path:
    """processed_manifest.csv -> processed_manifest.<tag>.csv（tag 为空则原样）。"""
    if not tag:
        return path
    return path.with_name(f"{path.stem}.{tag}{path.suffix}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess all sessions from the manifest (eog_ecg_clean).")
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--config", default="configs/preprocess.yaml")
    ap.add_argument("--subjects", default=None, help="逗号分隔，如 '1,2' 或 'sub-001'；默认全部。")
    ap.add_argument("--sessions", default=None, help="逗号分隔，如 '1,2,3' 或 'ses-01'；默认全部。")
    ap.add_argument("--limit", type=int, default=None, help="过滤后只处理前 N 个 session（dry-run 用）。")
    ap.add_argument("--tag", default=None, help="给输出 manifest/summary 文件名加后缀（如 dryrun）。")
    args = ap.parse_args()

    P = paths.load_paths(PROJECT_ROOT / args.paths, require_raw=True)
    cfg = load_config(PROJECT_ROOT / args.config)
    mode = cfg.get("mode", cfg.get("variant", "eog_ecg_clean"))
    if mode != "eog_ecg_clean":
        raise ValueError(f"preprocess_all 仅支持 mode=eog_ecg_clean，当前 config mode={mode}。")

    # 输出根目录安全校验：拒绝写入 raw 的 sourcedata/derivatives；不可写直接报错。
    out_root = P.processed_dir(mode)
    paths.assert_safe_output_dir(out_root, P)
    paths.ensure_writable_dir(out_root)
    logger.info("mode=%s | 输出根: %s", mode, out_root)

    subj_filter = _parse_filter(args.subjects, paths.sub_id)
    sess_filter = _parse_filter(args.sessions, paths.ses_id)

    rows = read_manifest(P.raw_manifest)
    selected = []
    for r in rows:
        s, ss = r["subject_id"], r["session_id"]
        if subj_filter and s not in subj_filter:
            continue
        if sess_filter and ss not in sess_filter:
            continue
        selected.append(r)
    if args.limit is not None:
        selected = selected[: args.limit]
    logger.info("raw manifest %s -> 选中 %d/%d 个 session 处理。", P.raw_manifest, len(selected), len(rows))
    if not selected:
        raise RuntimeError("过滤后没有可处理的 session（检查 --subjects/--sessions/--limit）。")

    results = []
    t0 = time.time()
    n_ok = n_failed = 0
    for i, r in enumerate(selected, 1):
        subj, sess = r["subject_id"], r["session_id"]
        ts = time.time()
        try:
            row = process_session_eog_ecg_clean(P, cfg, subj, sess, mode=mode, logger=logger)
            n_ok += 1 if row.get("status") == "ok" else 0
            n_failed += 0 if row.get("status") == "ok" else 1
        except Exception as e:  # noqa: BLE001 - 单 session 失败不影响全量
            logger.exception("%s/%s 预处理失败：%s", subj, sess, e)
            row = build_processed_manifest_row(
                subject_id=subj, session_id=sess, status="failed", error_message=repr(e))
            n_failed += 1
        results.append(row)
        logger.info("[%d/%d] %s/%s 用时 %.1fs（ok=%d failed=%d）",
                    i, len(selected), subj, sess, time.time() - ts, n_ok, n_failed)

    # 汇总：processed_manifest.csv + preprocess_summary.csv（写在 npz 输出根目录旁）。
    manifest_path = _tagged(P.processed_manifest, args.tag)
    summary_path = _tagged(out_root / "preprocess_summary.csv", args.tag)
    write_manifest_csv(results, manifest_path, fieldnames=PROCESSED_MANIFEST_FIELDS)
    summary_rows = [{k: row.get(k, "") for k in SUMMARY_FIELDS} for row in results]
    write_manifest_csv(summary_rows, summary_path, fieldnames=SUMMARY_FIELDS)

    logger.info("完成：%d ok / %d failed / 共 %d（用时 %.1fs）。",
                n_ok, n_failed, len(results), time.time() - t0)
    logger.info("processed manifest: %s", manifest_path)
    logger.info("summary: %s", summary_path)
    if n_failed:
        logger.warning("有 %d 个 session 失败，详见 manifest 的 error_message 列。", n_failed)


if __name__ == "__main__":
    main()
