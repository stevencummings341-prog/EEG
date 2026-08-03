#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SHU 2022 全量预处理：per-session .mat -> 统一 .npz + processed_manifest.csv。

依赖: numpy, scipy, pyyaml

数据入口为作者提供的 per-session `.mat`（见 code/preprocessing/shu_mat.py 说明）。
输出（默认写 workspace2，不写仓库；与 WBCIC processed 同惯例）:
  <out_root>/<sub>/<ses>/<sub>_<ses>_task-motorimagery_eeg.npz
  <out_root>/<sub>/<ses>/meta.json
  <out_root>/processed_manifest.csv

运行:
  python scripts/preprocess_shu.py --dry-run                 # 只枚举，不落盘
  python scripts/preprocess_shu.py --subjects 1              # 单被试
  python scripts/preprocess_shu.py                           # 全量 (25 人 × 5 session)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

CODE_ROOT = Path(__file__).resolve().parents[1] / "code"
PROJECT_ROOT = CODE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.datasets.manifest import PROCESSED_MANIFEST_FIELDS, write_manifest_csv  # noqa: E402
from code.preprocessing.shu_mat import load_shu_session_mat  # noqa: E402
from code.utils.io import save_json, save_session_npz  # noqa: E402

_PATTERN = re.compile(r"^(sub-\d+)_+(ses-\d+)_task_motorimagery_eeg\.mat$")
DEFAULT_OUT_ROOT = "/share/workspace2/moto_imagination/SHU/processed/npz_clean"


def _norm_sub(s: str) -> str:
    """'1' / 'sub-1' / 'sub-001' -> 'sub-001'。"""
    s = str(s).strip()
    if s.startswith("sub-"):
        s = s[4:]
    return f"sub-{int(s):03d}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess SHU 2022 .mat -> npz + manifest")
    ap.add_argument("--dataset-config",
                    default=str(CODE_ROOT / "configs" / "datasets" / "shu.yaml"))
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--subjects", help="逗号分隔，如 1,2 或 sub-001")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.dataset_config).read_text(encoding="utf-8"))
    data_dir = Path(cfg["data_dir"])
    channels = list(cfg.get("channels", []))
    sfreq = int(float(cfg.get("sfreq", 250.0)))
    n_ch = int(cfg.get("n_channels", 32))
    n_times = int(cfg.get("n_timepoints", 1000))
    out_root = Path(args.out_root)

    subj_filter = {_norm_sub(s) for s in args.subjects.split(",")} if args.subjects else None

    # 枚举 session
    sessions = []
    for p in sorted(data_dir.iterdir()):
        m = _PATTERN.match(p.name)
        if not m:
            continue
        sub, ses = m.group(1), m.group(2)
        sub = _norm_sub(sub)
        if subj_filter and sub not in subj_filter:
            continue
        sessions.append((sub, ses, p))

    if not sessions:
        raise SystemExit(f"未在 {data_dir} 找到匹配的 .mat（subjects={args.subjects}）。")

    print(f"[shu] {len(sessions)} sessions / {len({s for s,_,_ in sessions})} subjects "
          f"-> out_root={out_root}{' (DRY-RUN)' if args.dry_run else ''}")

    rows = []
    n_ok = n_failed = 0
    for sub, ses, mat_path in sessions:
        try:
            X, y, meta = load_shu_session_mat(mat_path, expect_channels=n_ch, expect_times=n_times)
        except Exception as e:  # noqa: BLE001 — 单 session 失败不中断全量
            print(f"  [ERR] {sub}/{ses}: {e}")
            rows.append({"subject_id": sub, "session_id": ses, "npz_path": "", "meta_path": "",
                         "report_path": "", "status": "failed", "n_trials": "", "n_channels": "",
                         "n_times": "", "sfreq": sfreq, "label_0_count": "", "label_1_count": "",
                         "labels_match_mat": "", "labels_multiset_match": "", "n_labels_agree": "",
                         "aux_cleaning_used": False, "valid_eog_channels": "", "valid_ecg_channels": "",
                         "ica_excluded_components": "", "error_message": str(e)})
            n_failed += 1
            continue

        status = meta["status"]
        npz_path = meta_path = ""
        if not args.dry_run:
            out_dir = out_root / sub / ses
            out_dir.mkdir(parents=True, exist_ok=True)
            npz_path = str(save_session_npz(
                out_dir / f"{sub}_{ses}_task-motorimagery_eeg.npz", X, y,
                subject_id=sub, session_id=ses, sfreq=sfreq, channel_names=channels, compress=True))
            meta_path = str(out_dir / "meta.json")
            save_json(meta, Path(meta_path))

        rows.append({
            "subject_id": sub, "session_id": ses, "npz_path": npz_path, "meta_path": meta_path,
            "report_path": "", "status": status, "n_trials": meta["n_trials"],
            "n_channels": meta["n_channels"], "n_times": meta["n_times"], "sfreq": sfreq,
            "label_0_count": meta["label_0_count"], "label_1_count": meta["label_1_count"],
            "labels_match_mat": "", "labels_multiset_match": "", "n_labels_agree": "",
            "aux_cleaning_used": False, "valid_eog_channels": "", "valid_ecg_channels": "",
            "ica_excluded_components": "", "error_message": "; ".join(meta.get("fail_reasons", []))})
        n_ok += int(status == "ok")
        n_failed += int(status != "ok")
        print(f"  {sub}/{ses} shape={meta['output_shape']} labels={meta['label_counts']} status={status}")

    if not args.dry_run:
        manifest_path = out_root / "processed_manifest.csv"
        write_manifest_csv(rows, manifest_path, fieldnames=PROCESSED_MANIFEST_FIELDS)
        print(f"[shu] manifest -> {manifest_path}")
    print(f"[shu] DONE: ok={n_ok} failed={n_failed}")


if __name__ == "__main__":
    main()
