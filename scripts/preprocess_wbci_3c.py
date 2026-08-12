#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""WBCIC-SHU 2025 三分类：官方 derivatives .mat -> 统一 .npz + processed_manifest.csv。

输出默认写到项目内（外部 processed/ 对当前机只读）:
  outputs/processed/wbci_shu_3c_mat_clean/<sub>/<ses>/*.npz
  outputs/processed/wbci_shu_3c_mat_clean/processed_manifest.csv

运行:
  python scripts/preprocess_wbci_3c.py --dry-run
  python scripts/preprocess_wbci_3c.py
  python scripts/preprocess_wbci_3c.py --subjects 1,2

路径: 3C mat 根来自 paths.local.yaml 的 raw_data.shu_2c_root + derivatives 3C 子目录；
输出根来自 manifests.wbci_3c_processed_manifest 的父目录，或 --out-root。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1] / "code"
PROJECT_ROOT = CODE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.datasets.manifest import PROCESSED_MANIFEST_FIELDS, write_manifest_csv  # noqa: E402
from code.preprocessing.wbci_3c_mat import load_wbci_3c_session_mat  # noqa: E402
from code.utils.io import save_json, save_session_npz  # noqa: E402
from code.utils.paths import load_paths, sub_id, ses_id  # noqa: E402

# Same 58 EEG names as 2C eog_ecg_clean npz (Pz dropped in official 58ch derivative).
WBCI_58_CHANNELS = [
    "Fpz", "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8",
    "Fz", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8",
    "FCz", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FT7", "FT8",
    "Cz", "C1", "C2", "C3", "C4", "C5", "C6", "T7", "T8",
    "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "TP7", "TP8",
    "P3", "P4", "P5", "P6", "P7", "P8",
    "POz", "PO3", "PO4", "PO5", "PO6", "PO7", "PO8",
    "Oz", "O1", "O2",
]


def _default_3c_mat_root(raw_root: Path) -> Path:
    return raw_root / "derivatives" / "3C dataset_processeddata"


def _enumerate_sessions(mat_root: Path, subj_filter: set[str] | None):
    sessions = []
    for sub_dir in sorted(mat_root.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        sub = sub_id(sub_dir.name)
        if subj_filter and sub not in subj_filter:
            continue
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            ses = ses_id(ses_dir.name)
            mats = list((ses_dir / "eeg").glob("*_task-motorimagery_eeg.mat"))
            if not mats:
                mats = list(ses_dir.rglob("*_task-motorimagery_eeg.mat"))
            if not mats:
                continue
            sessions.append((sub, ses, mats[0]))
    return sessions


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess WBCIC-SHU 3C .mat -> npz + manifest")
    ap.add_argument("--mat-root", default=None,
                    help="3C derivatives root (default: <raw>/derivatives/3C dataset_processeddata)")
    ap.add_argument("--out-root", default=None,
                    help="output root (default: outputs/processed/wbci_shu_3c_mat_clean)")
    ap.add_argument("--subjects", help="逗号分隔，如 1,2 或 sub-001")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    P = load_paths(require_raw=False)
    mat_root = Path(args.mat_root) if args.mat_root else _default_3c_mat_root(P.raw_root)
    out_root = Path(args.out_root) if args.out_root else (
        PROJECT_ROOT / "outputs" / "processed" / "wbci_shu_3c_mat_clean"
    )
    if not mat_root.exists():
        raise SystemExit(f"3C mat root 不存在: {mat_root}")

    subj_filter = {sub_id(s) for s in args.subjects.split(",")} if args.subjects else None
    sessions = _enumerate_sessions(mat_root, subj_filter)
    if not sessions:
        raise SystemExit(f"未在 {mat_root} 找到 3C .mat（subjects={args.subjects}）。")

    print(f"[wbci_3c] {len(sessions)} sessions / {len({s for s, _, _ in sessions})} subjects "
          f"mat_root={mat_root} -> out_root={out_root}{' (DRY-RUN)' if args.dry_run else ''}")

    # Extend write schema with label_2_count while keeping legacy readers happy.
    fields = list(PROCESSED_MANIFEST_FIELDS)
    if "label_2_count" not in fields:
        fields.insert(fields.index("label_1_count") + 1, "label_2_count")

    rows = []
    n_ok = n_failed = 0
    for sub, ses, mat_path in sessions:
        try:
            X, y, meta = load_wbci_3c_session_mat(mat_path)
        except Exception as e:  # noqa: BLE001
            print(f"  [ERR] {sub}/{ses}: {e}")
            rows.append({
                "subject_id": sub, "session_id": ses, "npz_path": "", "meta_path": "",
                "report_path": "", "status": "failed", "n_trials": "", "n_channels": "",
                "n_times": "", "sfreq": 250, "label_0_count": "", "label_1_count": "",
                "label_2_count": "", "labels_match_mat": "", "labels_multiset_match": "",
                "n_labels_agree": "", "aux_cleaning_used": False, "valid_eog_channels": "",
                "valid_ecg_channels": "", "ica_excluded_components": "",
                "error_message": str(e),
            })
            n_failed += 1
            continue

        status = meta["status"]
        npz_path = meta_path = ""
        if not args.dry_run and status == "ok":
            out_dir = out_root / sub / ses
            out_dir.mkdir(parents=True, exist_ok=True)
            npz_path = str(save_session_npz(
                out_dir / f"{sub}_{ses}_task-motorimagery_eeg.npz", X, y,
                subject_id=sub, session_id=ses, sfreq=250.0,
                channel_names=WBCI_58_CHANNELS, compress=True))
            meta_path = str(out_dir / "meta.json")
            save_json(meta, Path(meta_path))

        rows.append({
            "subject_id": sub, "session_id": ses, "npz_path": npz_path, "meta_path": meta_path,
            "report_path": "", "status": status, "n_trials": meta["n_trials"],
            "n_channels": meta["n_channels"], "n_times": meta["n_times"], "sfreq": 250,
            "label_0_count": meta["label_0_count"], "label_1_count": meta["label_1_count"],
            "label_2_count": meta.get("label_2_count", 0),
            "labels_match_mat": "", "labels_multiset_match": "", "n_labels_agree": "",
            "aux_cleaning_used": False, "valid_eog_channels": "", "valid_ecg_channels": "",
            "ica_excluded_components": "",
            "error_message": "; ".join(meta.get("fail_reasons", [])),
        })
        n_ok += int(status == "ok")
        n_failed += int(status != "ok")
        print(f"  {sub}/{ses} shape={meta['output_shape']} labels={meta['label_counts']} status={status}")

    # When --subjects filters, merge into an existing full manifest instead of
    # wiping other subjects' rows (otherwise a one-subject rerun destroys the CSV).
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        manifest_path = out_root / "processed_manifest.csv"
        if subj_filter and manifest_path.exists():
            import csv
            with open(manifest_path, newline="", encoding="utf-8") as f:
                old = list(csv.DictReader(f))
            keep = [r for r in old if sub_id(r.get("subject_id", "")) not in subj_filter]
            rows = keep + rows
            # stable order
            rows.sort(key=lambda r: (r.get("subject_id", ""), r.get("session_id", "")))
        write_manifest_csv(rows, manifest_path, fieldnames=fields)
        print(f"[wbci_3c] manifest -> {manifest_path}")
    print(f"[wbci_3c] DONE: ok={n_ok} failed={n_failed}")


if __name__ == "__main__":
    main()
