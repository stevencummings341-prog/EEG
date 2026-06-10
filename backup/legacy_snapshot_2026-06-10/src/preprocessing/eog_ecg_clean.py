"""SHU 2C — EOG/ECG 辅助降噪预处理（正式版，mode=eog_ecg_clean）。

流程：
  1. 读 64 通道 data.bdf。
  2. 按真实通道名识别 EEG / ECG / EOG(HEOR/HEOL/VEOU/VEOL)。
     不依赖 task-motorimagery_eeg.json 的 EOG/ECG 计数（已确认是反的）。
  3. 校验每个辅助通道有效性：全 0 / 平线（std、峰峰值过小）/ NaN、Inf。
  4. 设置通道类型（eog/ecg/eeg）。
  5. 若启用且存在有效辅助通道：MNE ICA 清理
       - 在 1 Hz 高通副本上仅拟合 EEG；
       - 用有效 EOG 通道 find_bads_eog、有效 ECG 通道 find_bads_ecg；
       - 排除命中的 components，apply 回 EEG raw；
       - 记录 excluded ids / 相关分数 / 用到的辅助通道 / 警告。
     ICA/检测失败不静默：记录原因并退化为 no-aux clean（不抛出，由编排脚本汇总）。
  6. 论文式后半：删辅助通道 -> 重参考 Pz 再删 Pz(58 EEG) -> 0.5-40 带通 + 50 notch
     -> 用 evt.bdf TAL 提取 200 trigger（1->0, 2->1）-> [0,4)s 切段（整段去均值）
     -> 250 Hz 重采样。
  7. 输出 X=[200,58,1000] float32(µV)、y=[200]∈{0,1}、meta、report（含质检与 aux 信息）。

返回 (X, y, meta, report)。strict 下得不到 [200,58,1000] 抛错（编排脚本捕获并记录）。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .neuracle_events import read_neuracle_tals


def _channel_stats(data_uv: np.ndarray) -> Dict[str, Any]:
    """单通道统计量（输入单位 µV）。"""
    finite = np.isfinite(data_uv)
    has_nan = bool(np.isnan(data_uv).any())
    has_inf = bool(np.isinf(data_uv).any())
    vals = data_uv[finite]
    if vals.size == 0:
        return {"std": 0.0, "ptp": 0.0, "all_zero": True,
                "has_nan": has_nan, "has_inf": has_inf}
    return {
        "std": float(np.std(vals)),
        "ptp": float(np.ptp(vals)),
        "all_zero": bool(np.all(vals == 0.0)),
        "has_nan": has_nan,
        "has_inf": has_inf,
    }


def _check_aux_validity(raw, names: List[str], vcfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """逐个辅助通道判定有效性。返回 {name: {valid, reasons, std, ptp, ...}}。"""
    min_std = float(vcfg.get("min_std_uv", 1e-3))
    min_ptp = float(vcfg.get("min_ptp_uv", 1e-2))
    out: Dict[str, Dict[str, Any]] = {}
    for name in names:
        if name not in raw.ch_names:
            out[name] = {"present": False, "valid": False, "reasons": ["channel_absent"]}
            continue
        data = raw.get_data(picks=[name])[0]            # µV 量级
        st = _channel_stats(data)
        reasons: List[str] = []
        if st["has_nan"]:
            reasons.append("nan")
        if st["has_inf"]:
            reasons.append("inf")
        if st["all_zero"]:
            reasons.append("all_zero")
        if st["std"] < min_std:
            reasons.append(f"std<{min_std}")
        if st["ptp"] < min_ptp:
            reasons.append(f"ptp<{min_ptp}")
        out[name] = {
            "present": True,
            "valid": len(reasons) == 0,
            "reasons": reasons,
            "std": round(st["std"], 6),
            "ptp": round(st["ptp"], 6),
        }
    return out


def _scores_per_component(scores: Any) -> np.ndarray:
    """把 find_bads_* 的分数折叠成每分量一个绝对值（多辅助通道取各分量的最大 |score|）。"""
    if scores is None:
        return np.array([])
    if isinstance(scores, (list, tuple)):
        mats = [np.abs(np.asarray(s, dtype=float)).ravel() for s in scores]
        return np.max(np.vstack(mats), axis=0) if mats else np.array([])
    arr = np.abs(np.asarray(scores, dtype=float))
    return arr if arr.ndim == 1 else np.max(arr, axis=0)


def _topk_scores(per_comp: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
    """返回 |score| 最大的前 k 个分量 [{component, score}]（降序），用于报告/诊断。"""
    if per_comp.size == 0:
        return []
    order = np.argsort(per_comp)[::-1][:k]
    return [{"component": int(i), "score": round(float(per_comp[i]), 4)} for i in order]


def _default_aux_info() -> Dict[str, Any]:
    return {
        "aux_cleaning_used": False,
        "method": None,
        "fallback_used": None,
        "valid_eog_channels": [],
        "valid_ecg_channels": [],
        "ica_n_components": None,
        "ica_excluded_components": [],
        "eog_excluded_components": [],
        "ecg_excluded_components": [],
        "eog_component_scores": [],     # 最强的几个 EOG 相关分量（|corr|，含未排除的）
        "ecg_component_scores": [],     # 最强的几个 ECG 相关分量（|corr|，含未排除的）
        "cleaning_warnings": [],
        "aux_validity": {},
    }


def _run_ica_clean(raw, valid_eog: List[str], valid_ecg: List[str],
                   icfg: Dict[str, Any], info: Dict[str, Any]) -> None:
    """在 raw 上原地做 ICA 清理；命中的分量写入 info。出错由调用方捕获。"""
    import mne

    hp = float(icfg.get("highpass_for_fit", 1.0))
    raw_fit = raw.copy()
    raw_fit.filter(l_freq=hp, h_freq=None, picks="eeg", fir_design="firwin", verbose="ERROR")

    n_comp = icfg.get("n_components", 0.99)
    max_iter = icfg.get("max_iter", "auto")
    method = str(icfg.get("method", "fastica"))
    decim = icfg.get("decim", None)
    decim_val = int(decim) if decim else None

    def _make_ica(nc):
        return mne.preprocessing.ICA(
            n_components=nc, method=method,
            max_iter=max_iter if max_iter == "auto" else int(max_iter),
            random_state=int(icfg.get("random_state", 97)))

    # 拟合；若 float 方差比导致分量塌缩（如单一 PCA 分量占 ~99%，MNE 报错），
    # 用固定整数分量数重试一次，避免在高幅 session 上丢失辅助降噪。
    n_comp_used = n_comp
    try:
        ica = _make_ica(n_comp)
        ica.fit(raw_fit, picks="eeg", decim=decim_val, verbose="ERROR")
    except Exception as e:  # noqa: BLE001
        fb = int(icfg.get("n_components_fallback", 15))
        info["cleaning_warnings"].append(
            f"ICA fit (n_components={n_comp}) failed: {e!r}; 用 n_components={fb} 重试")
        ica = _make_ica(fb)
        ica.fit(raw_fit, picks="eeg", decim=decim_val, verbose="ERROR")
        n_comp_used = fb
    info["ica_n_components"] = int(ica.n_components_)
    info["ica_params"] = {
        "n_components_requested": n_comp,
        "n_components_used": n_comp_used,
        "method": method,
        "decim": decim_val,
        "highpass_for_fit": hp,
    }

    eog_measure = str(icfg.get("eog_measure", "correlation"))
    eog_thr = float(icfg.get("eog_threshold", 0.5))
    ecg_method = str(icfg.get("ecg_method", "correlation"))
    ecg_measure = str(icfg.get("ecg_measure", "correlation"))
    ecg_thr = icfg.get("ecg_threshold", 0.5)
    info["detection"] = {
        "eog_measure": eog_measure, "eog_threshold": eog_thr,
        "ecg_method": ecg_method, "ecg_measure": ecg_measure, "ecg_threshold": ecg_thr,
    }

    eog_idx: List[int] = []
    eog_scores = None
    if valid_eog:
        try:
            eog_idx, eog_scores = ica.find_bads_eog(
                raw_fit, ch_name=valid_eog, threshold=eog_thr,
                measure=eog_measure, verbose="ERROR")
        except Exception as e:  # noqa: BLE001 - 记录而非崩溃
            info["cleaning_warnings"].append(f"find_bads_eog failed: {e!r}")

    ecg_idx: List[int] = []
    ecg_scores = None
    if valid_ecg:
        ch = valid_ecg[0]
        try:
            ecg_idx, ecg_scores = ica.find_bads_ecg(
                raw_fit, ch_name=ch, method=ecg_method, threshold=ecg_thr,
                measure=ecg_measure, verbose="ERROR")
        except Exception as e:  # noqa: BLE001
            info["cleaning_warnings"].append(f"find_bads_ecg({ecg_method}) failed: {e!r}; trying ctps")
            try:
                ecg_idx, ecg_scores = ica.find_bads_ecg(
                    raw_fit, ch_name=ch, method="ctps", verbose="ERROR")
                info["cleaning_warnings"].append("find_bads_ecg fell back to ctps")
            except Exception as e2:  # noqa: BLE001
                info["cleaning_warnings"].append(f"find_bads_ecg(ctps) failed: {e2!r}")

    eog_idx = sorted({int(i) for i in eog_idx})
    ecg_idx = sorted({int(i) for i in ecg_idx})
    exclude = sorted(set(eog_idx) | set(ecg_idx))
    # 始终记录最强的几个相关分量（含未排除的），便于诊断「为何排/不排」。
    info["eog_component_scores"] = _topk_scores(_scores_per_component(eog_scores), k=5)
    info["ecg_component_scores"] = _topk_scores(_scores_per_component(ecg_scores), k=5)
    info["eog_excluded_components"] = eog_idx
    info["ecg_excluded_components"] = ecg_idx
    info["ica_excluded_components"] = exclude
    if not exclude:
        info["cleaning_warnings"].append("ICA 拟合成功但未检出超阈值的 EOG/ECG 相关分量；EEG 未改动。")

    ica.exclude = exclude
    ica.apply(raw, verbose="ERROR")          # 仅重建 EEG 通道；辅助通道不变
    info["aux_cleaning_used"] = True
    info["method"] = "ica"


def _quality_checks(X: np.ndarray, y: np.ndarray, sfreq: int,
                    expect_shape: Tuple[int, int, int], expected_per_class: int) -> Dict[str, Any]:
    counts = Counter(y.tolist())
    n0, n1 = int(counts.get(0, 0)), int(counts.get(1, 0))
    has_nan = bool(np.isnan(X).any())
    has_inf = bool(np.isinf(X).any())
    checks = {
        "shape": list(X.shape),
        "shape_ok": tuple(X.shape) == tuple(expect_shape),
        "y_shape": list(y.shape),
        "y_shape_ok": tuple(y.shape) == (expect_shape[0],),
        "X_dtype": str(X.dtype),
        "X_dtype_ok": X.dtype == np.float32,
        "y_dtype": str(y.dtype),
        "y_dtype_ok": y.dtype == np.int64,
        "n_channels": int(X.shape[1]),
        "n_channels_ok": int(X.shape[1]) == int(expect_shape[1]),
        "n_times": int(X.shape[2]),
        "n_times_ok": int(X.shape[2]) == int(expect_shape[2]),
        "sfreq": int(sfreq),
        "sfreq_ok": int(sfreq) == 250,
        "label_0_count": n0,
        "label_1_count": n1,
        "label_balance_ok": (n0 == expected_per_class and n1 == expected_per_class),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "no_nan_inf": (not has_nan) and (not has_inf),
    }
    checks["all_passed"] = bool(
        checks["shape_ok"] and checks["y_shape_ok"] and checks["X_dtype_ok"]
        and checks["y_dtype_ok"] and checks["n_channels_ok"] and checks["n_times_ok"]
        and checks["sfreq_ok"] and checks["label_balance_ok"] and checks["no_nan_inf"]
    )
    return checks


def evaluate_failure_reasons(
    qc: Dict[str, Any],
    n_events: int,
    mat_crosscheck: Optional[Dict[str, Any]] = None,
    expected_trials: int = 200,
    expected_per_class: int = 100,
) -> List[str]:
    """判定一个 session 是否失败，返回失败原因列表（空 = 通过）。

    硬失败条件：形状不是 [200,58,1000]、label 不是 100/100、trigger 数不是 200、
    含 NaN/Inf、或（有 .mat 时）labels_multiset_match=False。
    注意：labels_match_mat=False（仅试次顺序与论文 .mat 不同）本身【不】判失败。
    """
    reasons: List[str] = []
    if not qc.get("shape_ok"):
        reasons.append(f"shape {qc.get('shape')} != [{expected_trials},58,1000]")
    if qc.get("label_0_count") != expected_per_class or qc.get("label_1_count") != expected_per_class:
        reasons.append(
            f"label_count {qc.get('label_0_count')}/{qc.get('label_1_count')}"
            f" != {expected_per_class}/{expected_per_class}")
    if int(n_events) != int(expected_trials):
        reasons.append(f"trigger_count {n_events} != {expected_trials}")
    if qc.get("has_nan") or qc.get("has_inf"):
        reasons.append("nan_or_inf_in_X")
    if mat_crosscheck and mat_crosscheck.get("checked") and mat_crosscheck.get("labels_multiset_match") is False:
        reasons.append("labels_multiset_match=False")
    return reasons


def preprocess_one_session_eog_ecg_clean(
    data_bdf: str | Path,
    evt_bdf: str | Path,
    config: Dict[str, Any],
    subject: Optional[str] = None,
    session: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """对单 session 做 EOG/ECG 辅助降噪预处理，返回 (X, y, meta, report)。"""
    import mne

    ch_cfg = config["channels"]
    fl_cfg = config["filter"]
    ep_cfg = config["epoch"]
    rs_cfg = config["resample"]
    aux_cfg = config.get("aux_cleaning", {}) or {}

    target_sfreq = int(rs_cfg["target_sfreq"])
    event_values = [int(v) for v in ep_cfg["event_values"]]
    label_map = {int(k): int(v) for k, v in ep_cfg["label_map"].items()}
    expect_shape = tuple(int(v) for v in config["expect_shape"])         # (200, 58, 1000)
    n_times_target = int(expect_shape[2])
    expected_per_class = int(ep_cfg.get("expected_per_class", expect_shape[0] // 2))
    strict = bool(config.get("strict", True))

    eog_names = list(ch_cfg.get("eog_names", []))
    ecg_names = list(ch_cfg.get("ecg_names", []))
    ref = ch_cfg["reference"]

    # --- 1. 读 64 通道 raw ---
    raw = mne.io.read_raw_bdf(str(data_bdf), preload=True, verbose="ERROR")
    raw_sfreq = float(raw.info["sfreq"])
    orig_nchan = len(raw.ch_names)

    # --- 2. 按真实名识别 EEG/EOG/ECG（EEG = 其余非辅助通道）---
    aux_all = [c for c in (ecg_names + eog_names) if c in raw.ch_names]
    eeg_names = [c for c in raw.ch_names if c not in aux_all]

    # --- 3. 校验辅助通道有效性 ---
    aux_info = _default_aux_info()
    validity = _check_aux_validity(raw, ecg_names + eog_names, aux_cfg.get("validity", {}))
    aux_info["aux_validity"] = validity
    valid_eog = [c for c in eog_names if validity.get(c, {}).get("valid")]
    valid_ecg = [c for c in ecg_names if validity.get(c, {}).get("valid")]
    aux_info["valid_eog_channels"] = valid_eog
    aux_info["valid_ecg_channels"] = valid_ecg

    # --- 4. 设置通道类型（含无效辅助通道也设类型，确保 picks="eeg" 恰为 EEG）---
    type_map: Dict[str, str] = {}
    for c in eog_names:
        if c in raw.ch_names:
            type_map[c] = "eog"
    for c in ecg_names:
        if c in raw.ch_names:
            type_map[c] = "ecg"
    for c in eeg_names:
        type_map[c] = "eeg"
    raw.set_channel_types(type_map, verbose="ERROR")

    # --- 5. ICA 清理（启用、方法=ica、且至少有一个有效辅助通道时）---
    aux_enabled = bool(aux_cfg.get("enabled", True))
    method = str(aux_cfg.get("method", "ica"))
    if not aux_enabled:
        aux_info["cleaning_warnings"].append("aux_cleaning.enabled=false：跳过辅助降噪。")
        aux_info["fallback_used"] = "disabled"
    elif method != "ica":
        aux_info["cleaning_warnings"].append(f"未实现的 aux_cleaning.method={method}：退化为 no_aux_clean。")
        aux_info["fallback_used"] = aux_cfg.get("fallback", "no_aux_clean")
    elif not (valid_eog or valid_ecg):
        aux_info["cleaning_warnings"].append("没有有效的 EOG/ECG 辅助通道：退化为 no_aux_clean。")
        aux_info["fallback_used"] = aux_cfg.get("fallback", "no_aux_clean")
    else:
        try:
            _run_ica_clean(raw, valid_eog, valid_ecg, aux_cfg.get("ica", {}) or {}, aux_info)
        except Exception as e:  # noqa: BLE001 - ICA 整体失败也不崩，退化处理
            aux_info["aux_cleaning_used"] = False
            aux_info["fallback_used"] = aux_cfg.get("fallback", "no_aux_clean")
            aux_info["cleaning_warnings"].append(f"ICA cleaning failed -> fallback no_aux_clean: {e!r}")

    # --- 6. 论文式后半 ---
    # 6a. 删辅助通道 -> 59 EEG。
    dropped = [c for c in aux_all if c in raw.ch_names]
    raw.drop_channels(dropped)
    raw.set_channel_types({c: "eeg" for c in raw.ch_names}, verbose="ERROR")

    # 6b. 重参考 Pz 再删 Pz -> 58 EEG。
    if ref not in raw.ch_names:
        raise ValueError(f"参考通道 {ref} 不在数据中: {raw.ch_names}")
    raw.set_eeg_reference(ref_channels=[ref], verbose="ERROR")
    raw.drop_channels([ref])
    final_ch_names = list(raw.ch_names)
    final_nchan = len(final_ch_names)

    # 6c. 带通 + 陷波。
    raw.filter(fl_cfg["l_freq"], fl_cfg["h_freq"], fir_design="firwin", verbose="ERROR")
    if fl_cfg.get("notch_freq"):
        raw.notch_filter(freqs=[float(fl_cfg["notch_freq"])], verbose="ERROR")

    # 6d. 事件（evt.bdf TAL）。
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

    # 6e. 切段 [0,4)（整段去均值）。
    tmin = float(ep_cfg["tmin"])
    win = float(ep_cfg["tmax"]) - tmin
    tmax = tmin + win - 1.0 / raw_sfreq
    event_id = {str(c): c for c in event_values}
    epochs = mne.Epochs(raw, mne_events, event_id=event_id, tmin=tmin, tmax=tmax,
                        baseline=(None, None), preload=True, verbose="ERROR")

    # 6f. 重采样 250 Hz。
    epochs.resample(target_sfreq, verbose="ERROR")

    # --- 7. 取数据、质检 ---
    X = epochs.get_data().astype(np.float32)            # [n, ch, time]，µV
    if X.shape[2] > n_times_target:
        X = X[:, :, :n_times_target]
    y = np.array([label_map[int(c)] for c in epochs.events[:, 2]], dtype=np.int64)
    label_counts = {int(k): int(v) for k, v in Counter(y.tolist()).items()}

    qc = _quality_checks(X, y, target_sfreq, expect_shape, expected_per_class)
    # 不带 .mat 的硬检查（形状/标签数/trigger 数/NaN-Inf）；multiset 检查由编排脚本在
    # 取到 .mat 后补充并最终定 status（见 pipeline.process_session_eog_ecg_clean）。
    fail_reasons = evaluate_failure_reasons(
        qc, n_events, mat_crosscheck=None,
        expected_trials=int(expect_shape[0]), expected_per_class=expected_per_class)
    status = "ok" if not fail_reasons else "failed"

    meta = {
        "subject_id": subject,
        "session_id": session,
        "mode": config.get("mode", config.get("variant", "eog_ecg_clean")),
        "data_bdf": str(data_bdf),
        "evt_bdf": str(evt_bdf),
        "raw_sfreq": raw_sfreq,
        "target_sfreq": target_sfreq,
        "orig_n_channels": orig_nchan,
        "eeg_channels_detected": len(eeg_names),
        "eog_names": eog_names,
        "ecg_names": ecg_names,
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
        "npz_keys": ["X", "y", "subject_id", "session_id", "sfreq", "channel_names"],
        "mne_version": mne.__version__,
        "aux_cleaning_used": aux_info["aux_cleaning_used"],
        "status": status,
    }
    if fail_reasons:
        meta["fail_reasons"] = fail_reasons

    report = {
        "subject_id": subject,
        "session_id": session,
        "mode": meta["mode"],
        "status": status,
        "fail_reasons": fail_reasons,
        "quality_checks": qc,
        "aux_cleaning": aux_info,
        # 下面三项由编排脚本在有 .mat 时填充（见 pipeline）：
        "labels_match_mat": None,        # 精确顺序是否一致（False 不判失败）
        "labels_multiset_match": None,   # 标签多重集是否一致（False 判失败）
        "n_labels_agree": None,
        "mat_crosscheck": None,
    }
    # 注：失败不在此抛出。形状/标签等失败由 status+fail_reasons 记录，编排脚本汇总到
    # summary（单 session 失败不影响全量）。仅“0 个 MI 事件 / 缺参考通道”等不可恢复
    # 情况才在前面抛异常。strict 仅作语义保留，不再强制 raise。
    _ = strict
    return X, y, meta, report
