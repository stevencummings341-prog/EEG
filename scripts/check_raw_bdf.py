#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查单个 subject/session 的原始 BDF 文件（Task 1）。

功能（见 00-project-context / 10-data-preprocessing 规则）：
  1. 用 MNE 读取 data.bdf（默认 preload=False，登录节点安全）。
  2. 尝试从 evt.bdf / annotations 读取事件。
  3. 打印并保存：采样率、通道数、通道名、数据时长、事件数量。
  4. 标记 EEG / ECG / EOG 通道（按 channels.tsv 中的 EEG 名单分类）。
  5. 仅载入一小段数据，检查 ECG/EOG 通道是否含非零信号。
  6. 输出 JSON 报告到 outputs/raw_check/。

不做训练，不建模型。只读数据集，绝不写入数据集目录。

用法（在项目根目录运行）：
  python scripts/check_raw_bdf.py --subject 1 --session 1
  python scripts/check_raw_bdf.py --data-bdf /path/data.bdf --evt-bdf /path/evt.bdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# 允许从项目根目录导入 src.*（脚本默认 sys.path[0] 是 scripts/，需补上根目录）。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import paths  # noqa: E402
from src.utils.io import save_json  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402

logger = get_logger("check_raw_bdf")


def load_eeg_channel_names(dataset_root: Path) -> list[str]:
    """从数据集的 task-motorimagery_channels.tsv 读取 59 个 EEG 通道名。

    读不到时返回空列表（后续按通道类型/名称启发式分类，并在报告里标注）。
    """
    tsv = dataset_root / "task-motorimagery_channels.tsv"
    names: list[str] = []
    try:
        with open(tsv, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            name_idx = header.index("name") if "name" in header else 0
            type_idx = header.index("type") if "type" in header else None
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if not parts or not parts[0]:
                    continue
                is_eeg = (type_idx is None) or (parts[type_idx].upper() == "EEG")
                if is_eeg:
                    names.append(parts[name_idx])
    except FileNotFoundError:
        logger.warning("未找到 channels.tsv: %s（将按名称启发式分类）", tsv)
    return names


def classify_channels(ch_names: list[str], eeg_names: list[str]) -> dict:
    """把通道分成 eeg / eog / ecg / other。

    优先用 channels.tsv 的 EEG 名单；非 EEG 的再按名字分类。
    本数据集已核实的辅助通道命名（见 docs/DATASET_SHU.md）：
      - ECG: "ECG"
      - EOG: HEOR/HEOL（水平）、VEOU/VEOL（垂直）—— 名字不含 "eog"，
        故按 H/V-EOG 前缀识别。
    """
    eeg_set = {n.lower() for n in eeg_names}
    roles = {"eeg": [], "eog": [], "ecg": [], "other": []}
    for ch in ch_names:
        low = ch.lower()
        if low in eeg_set:
            roles["eeg"].append(ch)
        elif "eog" in low or low.startswith(("heo", "veo")):
            # HEOR/HEOL/VEOU/VEOL 等水平/垂直眼电通道。
            roles["eog"].append(ch)
        elif "ecg" in low or "ekg" in low:
            roles["ecg"].append(ch)
        else:
            roles["other"].append(ch)
    return roles


def read_events(evt_bdf: Path, data_raw) -> dict:
    """尽力从 evt.bdf 与主文件读取事件，返回统计信息（不臆测 event id）。"""
    import mne

    info = {"evt_bdf_exists": evt_bdf.exists(), "annotations": {}, "methods_tried": []}

    # 方法 1：把 evt.bdf 当作带 annotations 的 BDF 读取。
    if evt_bdf.exists():
        info["methods_tried"].append("read_raw_bdf(evt.bdf).annotations")
        try:
            evt_raw = mne.io.read_raw_bdf(str(evt_bdf), preload=False, verbose="ERROR")
            ann = evt_raw.annotations
            descs, counts = np.unique(ann.description, return_counts=True) if len(ann) else ([], [])
            info["annotations"] = {
                "n_annotations": int(len(ann)),
                "by_description": {str(d): int(c) for d, c in zip(descs, counts)},
                "evt_channel_names": list(evt_raw.ch_names),
            }
        except Exception as e:  # noqa: BLE001 - 这里就是要把任何读取问题报告出来
            info["annotations"]["error"] = repr(e)

    # 方法 2：主文件自带 annotations。
    info["methods_tried"].append("data.bdf annotations")
    try:
        ann = data_raw.annotations
        if len(ann):
            descs, counts = np.unique(ann.description, return_counts=True)
            info["data_annotations"] = {
                "n_annotations": int(len(ann)),
                "by_description": {str(d): int(c) for d, c in zip(descs, counts)},
            }
    except Exception as e:  # noqa: BLE001
        info["data_annotations_error"] = repr(e)

    return info


def signal_activity(raw, picks: list[str], seconds: float = 10.0) -> dict:
    """只载入前 seconds 秒，检查给定通道是否含非零/非平坦信号。"""
    out = {}
    if not picks:
        return out
    sfreq = float(raw.info["sfreq"])
    stop = min(int(seconds * sfreq), raw.n_times)
    try:
        data = raw.get_data(picks=picks, start=0, stop=stop)  # [n_picks, n_samples]
        for ch, row in zip(picks, data):
            out[ch] = {
                "std": float(np.std(row)),
                "ptp": float(np.ptp(row)),
                "all_zero": bool(np.allclose(row, 0.0)),
                "has_nan": bool(np.isnan(row).any()),
            }
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect one raw BDF session (no training).")
    ap.add_argument("--config", default="configs/paths.yaml",
                    help="路径配置文件（含外部 raw 数据根）。")
    ap.add_argument("--subject", default=None, help="被试，如 1 或 sub-001。")
    ap.add_argument("--session", default=None, help="session，如 1 或 ses-01。")
    ap.add_argument("--data-bdf", default=None, help="直接指定 data.bdf 路径（绕过 paths.yaml）。")
    ap.add_argument("--evt-bdf", default=None, help="直接指定 evt.bdf 路径。")
    ap.add_argument("--seconds", type=float, default=10.0,
                    help="检查 ECG/EOG 活动时载入的秒数（默认 10s，登录节点安全）。")
    ap.add_argument("--out-dir", default="outputs/raw_check", help="JSON 报告输出目录。")
    args = ap.parse_args()

    import mne

    # 解析 data.bdf / evt.bdf 路径：优先显式路径，否则从 configs/paths.yaml 构造。
    root = None
    if args.data_bdf:
        data_bdf = Path(args.data_bdf)
        evt_bdf = Path(args.evt_bdf) if args.evt_bdf else data_bdf.with_name("evt.bdf")
        subj = args.subject or data_bdf.parent.parent.parent.name
        sess = args.session or data_bdf.parent.parent.name
        # 尝试拿到 raw 根，仅用于定位 channels.tsv（拿不到也能跑，通道分类退化为启发式）。
        try:
            root = paths.load_paths(PROJECT_ROOT / args.config, require_raw=False).raw_root
        except Exception:  # noqa: BLE001
            root = None
    else:
        if args.subject is None or args.session is None:
            ap.error("需要 --subject 与 --session，或直接给 --data-bdf。")
        P = paths.load_paths(PROJECT_ROOT / args.config, require_raw=True)
        root = P.raw_root
        subj = paths.sub_id(args.subject)
        sess = paths.ses_id(args.session)
        data_bdf, evt_bdf = P.raw_bdf_paths(subj, sess)

    logger.info("data.bdf = %s", data_bdf)
    logger.info("evt.bdf  = %s", evt_bdf)
    if not data_bdf.exists():
        raise FileNotFoundError(f"data.bdf 不存在: {data_bdf}")

    # 读取主文件头（preload=False，不把 432MB 全载入内存）。
    raw = mne.io.read_raw_bdf(str(data_bdf), preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    duration_s = float(raw.n_times) / sfreq

    eeg_names = load_eeg_channel_names(root) if root is not None else []
    roles = classify_channels(raw.ch_names, eeg_names)

    # 仅对疑似 ECG/EOG/other 通道做活动检查（这些是我们要确认是否可用的辅助通道）。
    aux_picks = roles["eog"] + roles["ecg"] + roles["other"]
    aux_activity = signal_activity(raw, aux_picks, seconds=args.seconds)

    events_info = read_events(evt_bdf, raw)

    report = {
        "subject": subj,
        "session": sess,
        "data_bdf": str(data_bdf),
        "evt_bdf": str(evt_bdf),
        "sfreq": sfreq,
        "n_channels": int(raw.info["nchan"]),
        "duration_seconds": duration_s,
        "n_times": int(raw.n_times),
        "channel_names": list(raw.ch_names),
        "channel_roles": roles,
        "n_eeg_detected": len(roles["eeg"]),
        "n_aux_detected": len(aux_picks),
        "aux_signal_activity": aux_activity,
        "events": events_info,
        "expected": {
            # 已核实的真实布局：59 EEG + 1 ECG("ECG") + 4 EOG(HEOR/HEOL/VEOU/VEOL)。
            # 注意 task-motorimagery_eeg.json 写的是 1 EOG + 4 ECG，与实际相反（已确认是元数据笔误）。
            "sfreq": 1000, "n_channels": 64, "n_eeg": 59, "n_ecg": 1, "n_eog": 4,
            "n_trials_per_session": 200,
            "note": "eeg.json 的 EOG/ECG 计数与实际通道名相反；以实际通道名为准。",
        },
        "mne_version": mne.__version__,
    }

    # 控制台摘要。
    logger.info("sfreq=%.1f Hz | n_channels=%d | duration=%.1f s",
                sfreq, report["n_channels"], duration_s)
    logger.info("detected EEG=%d, EOG=%s, ECG=%s, other=%s",
                len(roles["eeg"]), roles["eog"], roles["ecg"], roles["other"])
    if events_info.get("annotations", {}).get("by_description"):
        logger.info("evt.bdf annotations: %s", events_info["annotations"]["by_description"])
    else:
        logger.warning("evt.bdf 未解析出 annotations，请检查 events 字段。")

    out_dir = PROJECT_ROOT / args.out_dir
    out_path = out_dir / f"{subj}_{sess}_raw_check.json"
    save_json(report, out_path)
    logger.info("报告已保存: %s", out_path)


if __name__ == "__main__":
    main()
