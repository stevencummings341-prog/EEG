"""SHU 2C 数据集 paper-style 预处理。

严格对齐数据集作者的 MATLAB 配方 code/pre-processed/preprocessed.m：
  1. 读 data.bdf（64 通道 @1000Hz）。
  2. 从 evt.bdf 的 TAL 注释解析事件（见 neuracle_events，MNE 自带解析会漏）。
  3. 去掉 5 个辅助通道 {ECG, HEOR, HEOL, VEOU, VEOL} -> 59 EEG。
  4. 重参考到 Pz(EEGLAB 里是第 43 通道) 再去掉 Pz -> 58 EEG。
  5. 0.5-40Hz 带通；50Hz 陷波。
  6. 以事件为中心切 [0, 4) s 段（baseline 全段去均值，对应 pop_rmbase）。
  7. 降采样到 250Hz -> 每段 1000 点。
  8. 输出 X=[200,58,1000] float32(µV), y=[200]∈{0,1}, 以及 meta。

得不到 [200,58,1000] 时，strict 模式下抛错（绝不静默裁剪/补零）。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .neuracle_events import read_neuracle_tals


def preprocess_one_session(
    data_bdf: str | Path,
    evt_bdf: str | Path,
    config: Dict[str, Any],
    subject: str | None = None,
    session: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """对单个 subject/session 做 paper-style 预处理，返回 (X, y, meta)。"""
    import mne

    ch_cfg = config["channels"]
    fl_cfg = config["filter"]
    ep_cfg = config["epoch"]
    rs_cfg = config["resample"]

    target_sfreq = int(rs_cfg["target_sfreq"])
    event_values = [int(v) for v in ep_cfg["event_values"]]            # [1, 2]
    label_map = {int(k): int(v) for k, v in ep_cfg["label_map"].items()}  # {1:0, 2:1}
    expect_shape = tuple(config["expect_shape"])                        # (200, 58, 1000)
    n_times_target = int(expect_shape[2])
    strict = bool(config.get("strict", True))

    # --- 1. 读原始数据（preload 以便滤波/重参考）---
    raw = mne.io.read_raw_bdf(str(data_bdf), preload=True, verbose="ERROR")
    raw_sfreq = float(raw.info["sfreq"])
    orig_nchan = len(raw.ch_names)

    # --- 2. 解析事件（TAL）---
    events_list = []
    for onset, desc in read_neuracle_tals(evt_bdf):
        try:
            code = int(desc)
        except ValueError:
            continue
        if code in event_values:
            events_list.append((onset, code))
    events_list.sort(key=lambda x: x[0])
    n_events = len(events_list)
    if n_events == 0:
        raise ValueError(f"未从 {evt_bdf} 解析到任何 MI 事件（codes={event_values}）。")

    onsets = np.array([o for o, _ in events_list], dtype=float)
    codes = np.array([c for _, c in events_list], dtype=int)
    samples = np.round(onsets * raw_sfreq).astype(int)
    mne_events = np.column_stack([samples, np.zeros(n_events, int), codes])

    # --- 3. 去掉辅助通道（与文件中实际存在的交集）---
    aux_names = list(ch_cfg.get("ecg_names", [])) + list(ch_cfg.get("eog_names", []))
    dropped = [c for c in aux_names if c in raw.ch_names]
    raw.drop_channels(dropped)
    # 剩余通道统一标记为 eeg，确保 set_eeg_reference 可用（防止 MNE 把某些通道判成 misc）。
    raw.set_channel_types({c: "eeg" for c in raw.ch_names}, verbose="ERROR")

    # --- 4. 重参考到 Pz 后去掉 Pz ---
    ref = ch_cfg["reference"]
    if ref not in raw.ch_names:
        raise ValueError(f"参考通道 {ref} 不在数据中: {raw.ch_names}")
    raw.set_eeg_reference(ref_channels=[ref], verbose="ERROR")
    raw.drop_channels([ref])
    final_nchan = len(raw.ch_names)
    final_ch_names = list(raw.ch_names)

    # --- 5. 滤波：带通 + 陷波 ---
    raw.filter(fl_cfg["l_freq"], fl_cfg["h_freq"], fir_design="firwin",
               verbose="ERROR")
    if fl_cfg.get("notch_freq"):
        raw.notch_filter(freqs=[float(fl_cfg["notch_freq"])], verbose="ERROR")

    # --- 6. 切段：[tmin, tmin+win) ，整数采样点 ---
    tmin = float(ep_cfg["tmin"])
    win = float(ep_cfg["tmax"]) - tmin                 # 4.0 s
    tmax = tmin + win - 1.0 / raw_sfreq                # 取到 4000 点 @1000Hz
    event_id = {str(c): c for c in event_values}
    epochs = mne.Epochs(raw, mne_events, event_id=event_id, tmin=tmin, tmax=tmax,
                        baseline=(None, None), preload=True, verbose="ERROR")

    # --- 7. 降采样到 250Hz ---
    epochs.resample(target_sfreq, verbose="ERROR")

    # --- 8. 取数据并整理 ---
    # 单位说明（已核实）：BDF 头里 physical dim 是乱码 '?V'(本应是 µV)，MNE 不识别故
    # 不做 µV->V 换算，get_data() 直接返回 µV 量级数值。实测与论文 .mat 同尺度
    # （std 11.28 vs 11.26，相关 0.994），因此直接保存、不再乘 1e6。
    X = epochs.get_data().astype(np.float32)           # [n, ch, time]，单位 µV
    # 容错：resample 可能差 1 帧，多则裁剪，少则报错。
    if X.shape[2] > n_times_target:
        X = X[:, :, :n_times_target]
    y = np.array([label_map[int(c)] for c in epochs.events[:, 2]], dtype=np.int64)
    label_counts = {int(k): int(v) for k, v in Counter(y.tolist()).items()}

    status = "ok"
    fail_reason = None
    if X.shape != expect_shape:
        fail_reason = f"shape {X.shape} != expected {expect_shape}"
        status = "failed"

    meta = {
        "subject_id": subject,
        "session_id": session,
        "variant": config.get("variant", "paper_style"),
        "data_bdf": str(data_bdf),
        "evt_bdf": str(evt_bdf),
        "raw_sfreq": raw_sfreq,
        "target_sfreq": target_sfreq,
        "orig_n_channels": orig_nchan,
        "dropped_channels": dropped,
        "reference": ref,
        "final_n_channels": final_nchan,
        "final_ch_names": final_ch_names,
        "n_events": n_events,
        "n_trials": int(X.shape[0]),
        "label_counts": label_counts,
        "filter": {"l_freq": fl_cfg["l_freq"], "h_freq": fl_cfg["h_freq"],
                   "notch_freq": fl_cfg.get("notch_freq")},
        "epoch_window_s": [tmin, float(ep_cfg["tmax"])],
        "baseline": "whole-epoch demean (None, None)",
        "units": "uV",
        "output_shape": list(X.shape),
        "mne_version": mne.__version__,
        "status": status,
    }
    if fail_reason:
        meta["fail_reason"] = fail_reason

    if strict and status != "ok":
        raise ValueError(f"预处理结果不符合预期: {fail_reason}")

    return X, y, meta
