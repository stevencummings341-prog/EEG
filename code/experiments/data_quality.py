"""数据质量评估的可复用函数：幅值 / 频域 / 空间 / 配对相似度 / MI 可分性。

本模块只依赖 numpy + scipy，不依赖 torch，便于在质检脚本里独立调用。

约定：
  - 我们的 .npz：X [trials, 58, 1000] float32（µV），y [trials] int64（0/1）。
  - 官方 derivatives .mat：data [58,1000,200] float32，labels [1,200]∈{1,2}。
    读入后统一转成 X=[trials,58,1000]、y∈{0,1}，与我们的张量同构后再比较。

设计原则：每个函数输入/输出是纯 numpy；上层脚本负责 IO、CSV/JSON/图。
"""

from __future__ import annotations

import glob
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# numpy 2.x 把 trapz 改名为 trapezoid；两者都兼容。
_trapz = getattr(np, "trapezoid", np.trapz)

# --- 频带定义（Hz）。MI 主要看 mu/alpha 与 beta。---
EEG_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "mu_alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 40.0),
}
LINE_NOISE_BAND: Tuple[float, float] = (49.0, 51.0)

# 高幅 trial 判定阈值（µV，作用于每个 trial 在通道×时间上的 max|·|）。
HIGH_AMP_THRESHOLDS_UV: Tuple[float, ...] = (100.0, 200.0, 500.0)

# MI 相关通道（左右手运动想象的核心感觉运动区）。
MI_CHANNELS_DEFAULT: Tuple[str, ...] = ("C3", "C4", "Cz")


# ============================ 官方 .mat 定位与读取 ============================

def locate_official_mat(official_root: str | Path, subject: str, session: str) -> Optional[Path]:
    """在官方 derivatives 目录里定位某 session 的 .mat。

    官方目录命名不完全一致：多数在 ``sub/ses/eeg/<name>.mat``，个别直接在
    ``sub/ses/<name>.mat``（如 sub-001/ses-03）。这里两种都尝试，再退化为 glob。
    找不到返回 None。
    """
    root = Path(official_root)
    name = f"{subject}_{session}_task-motorimagery_eeg.mat"
    candidates = [
        root / subject / session / "eeg" / name,
        root / subject / session / name,
    ]
    for c in candidates:
        if c.exists():
            return c
    # 兜底：在该 session 目录下递归找任意 .mat。
    hits = sorted(glob.glob(str(root / subject / session / "**" / "*.mat"), recursive=True))
    return Path(hits[0]) if hits else None


def load_official_session(mat_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """读取官方 .mat，返回 (X[trials,58,1000] float32, y[trials] int64∈{0,1})。

    data [58,1000,200] -> transpose(2,0,1) -> [200,58,1000]；labels(1/2) -> 0/1。
    """
    import scipy.io as sio

    m = sio.loadmat(str(mat_path))
    if "data" not in m or "labels" not in m:
        raise KeyError(f"{mat_path} 缺少 data/labels（实际 keys={[k for k in m if not k.startswith('__')]}）")
    data = np.asarray(m["data"])
    if data.ndim != 3:
        raise ValueError(f"官方 data 维度异常: {data.shape}")
    X = np.ascontiguousarray(np.transpose(data, (2, 0, 1))).astype(np.float32)
    labels = np.asarray(m["labels"]).ravel().astype(np.int64)
    y = labels - 1  # 1/2 -> 0/1
    return X, y


# ============================ 标签一致性 ============================

def compare_labels(y_ours: np.ndarray, y_off: np.ndarray) -> Dict[str, Any]:
    """比较两组标签：精确顺序一致 / 多重集一致 / 逐位一致数。"""
    y_ours = np.asarray(y_ours).ravel().astype(np.int64)
    y_off = np.asarray(y_off).ravel().astype(np.int64)
    same_len = y_ours.shape[0] == y_off.shape[0]
    n_agree = int(np.sum(y_ours == y_off)) if same_len else 0
    exact = bool(same_len and n_agree == y_ours.shape[0])
    multiset = bool(
        np.array_equal(np.sort(y_ours), np.sort(y_off))
        if y_ours.shape[0] == y_off.shape[0]
        else False
    )
    return {
        "labels_match_exact": exact,
        "labels_multiset_match": multiset,
        "n_labels_agree": n_agree,
        "n_labels_ours": int(y_ours.shape[0]),
        "n_labels_official": int(y_off.shape[0]),
    }


# ============================ 幅值指标 ============================

def amplitude_metrics(X: np.ndarray) -> Dict[str, float]:
    """基础幅值指标（输入 [trials,ch,time]，单位 µV）。

    含 NaN/Inf 计数；其余统计在有限值上计算，避免被 NaN 污染。
    """
    X = np.asarray(X, dtype=np.float64)
    nan_count = int(np.isnan(X).sum())
    inf_count = int(np.isinf(X).sum())
    finite = X[np.isfinite(X)]
    if finite.size == 0:
        keys = ["mean", "std", "rms", "median_abs", "peak_to_peak", "max_abs",
                "trial_std_mean", "trial_std_median",
                "high_amp_trial_ratio_100uV", "high_amp_trial_ratio_200uV",
                "high_amp_trial_ratio_500uV"]
        out = {k: float("nan") for k in keys}
        out["nan_count"] = nan_count
        out["inf_count"] = inf_count
        return out

    # 逐 trial-通道的 peak-to-peak 与逐 trial std，对极端值更稳健。
    Xn = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    ptp_tc = np.ptp(Xn, axis=-1)                       # [trials, ch]
    trial_std = Xn.reshape(Xn.shape[0], -1).std(axis=1)  # [trials]
    per_trial_maxabs = np.abs(Xn).reshape(Xn.shape[0], -1).max(axis=1)  # [trials]
    n_trials = Xn.shape[0]

    out: Dict[str, float] = {
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "rms": float(np.sqrt(np.mean(finite ** 2))),
        "median_abs": float(np.median(np.abs(finite))),
        "peak_to_peak": float(ptp_tc.mean()),
        "max_abs": float(np.abs(finite).max()),
        "trial_std_mean": float(trial_std.mean()),
        "trial_std_median": float(np.median(trial_std)),
        "nan_count": nan_count,
        "inf_count": inf_count,
    }
    for thr in HIGH_AMP_THRESHOLDS_UV:
        out[f"high_amp_trial_ratio_{int(thr)}uV"] = float(np.mean(per_trial_maxabs > thr))
    return out


# ============================ 频域指标 ============================

def welch_psd(X: np.ndarray, sfreq: float, nperseg: int = 256,
              noverlap: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD（沿时间轴），返回 (freqs[nfreq], psd[..., nfreq])，单位 µV²/Hz。"""
    from scipy.signal import welch

    X = np.asarray(X, dtype=np.float64)
    nperseg = int(min(nperseg, X.shape[-1]))
    f, p = welch(X, fs=float(sfreq), nperseg=nperseg, noverlap=noverlap, axis=-1)
    return f, p


def bandpower(freqs: np.ndarray, psd: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    """对 psd 的最后一维在 [lo,hi] 频带积分，返回 psd.shape[:-1] 的功率。"""
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    if mask.sum() == 0:
        return np.zeros(psd.shape[:-1], dtype=np.float64)
    if mask.sum() < 2:
        return np.asarray(psd[..., mask]).mean(axis=-1)
    return _trapz(psd[..., mask], freqs[mask], axis=-1)


def band_powers(freqs: np.ndarray, psd_chmean: np.ndarray) -> Dict[str, float]:
    """对通道平均后的 PSD [nfreq] 计算各频带功率（含 50 Hz 工频）。"""
    out: Dict[str, float] = {}
    for name, band in EEG_BANDS.items():
        out[name] = float(bandpower(freqs, psd_chmean, band))
    out["line_noise_49_51"] = float(bandpower(freqs, psd_chmean, LINE_NOISE_BAND))
    return out


# ============================ 空间 / 通道指标 ============================

def per_channel_rms(X: np.ndarray) -> np.ndarray:
    """每通道 RMS [ch]（在 trials×time 上）。"""
    X = np.nan_to_num(np.asarray(X, dtype=np.float64))
    return np.sqrt(np.mean(X ** 2, axis=(0, 2)))


def per_channel_std(X: np.ndarray) -> np.ndarray:
    """每通道 std [ch]（在 trials×time 上）。"""
    X = np.nan_to_num(np.asarray(X, dtype=np.float64))
    return X.transpose(1, 0, 2).reshape(X.shape[1], -1).std(axis=1)


def _safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a / b
    r[~np.isfinite(r)] = np.nan
    return r


def per_channel_correlation(X_a: np.ndarray, X_b: np.ndarray) -> np.ndarray:
    """每通道相关（要求 trial 顺序一致、形状一致）。返回 [ch]。

    对每个通道，把 [trials,time] 展平后算 Pearson 相关。
    """
    if X_a.shape != X_b.shape:
        raise ValueError(f"per_channel_correlation 需要同形状: {X_a.shape} vs {X_b.shape}")
    A = np.nan_to_num(np.asarray(X_a, dtype=np.float64))
    B = np.nan_to_num(np.asarray(X_b, dtype=np.float64))
    ch = A.shape[1]
    a = A.transpose(1, 0, 2).reshape(ch, -1)
    b = B.transpose(1, 0, 2).reshape(ch, -1)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = np.sum(a * b, axis=1)
    den = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = num / den
    corr[~np.isfinite(corr)] = np.nan
    return corr


# ============================ 配对相似度（trial 顺序一致时） ============================

def paired_similarity(X_ours: np.ndarray, X_off: np.ndarray) -> Dict[str, float]:
    """trial 顺序一致时的配对相似度（trial-wise corr / per-channel corr / MAE / RMSE）。

    仅当两者形状完全一致时调用（labels_match_exact=True）。
    """
    if X_ours.shape != X_off.shape:
        raise ValueError(f"paired_similarity 需要同形状: {X_ours.shape} vs {X_off.shape}")
    A = np.nan_to_num(np.asarray(X_ours, dtype=np.float64))
    B = np.nan_to_num(np.asarray(X_off, dtype=np.float64))
    n = A.shape[0]

    # trial-wise Pearson：每个 trial 把 [ch,time] 展平后算相关。
    a = A.reshape(n, -1)
    b = B.reshape(n, -1)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = np.sum(a * b, axis=1)
    den = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        trial_corr = num / den
    trial_corr = trial_corr[np.isfinite(trial_corr)]

    ch_corr = per_channel_correlation(X_ours, X_off)
    diff = A - B
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    off_rms = float(np.sqrt(np.mean(B ** 2)))
    rel_rmse = float(rmse / off_rms) if off_rms > 0 else float("nan")

    return {
        "n_trials": int(n),
        "mean_trial_corr": float(np.mean(trial_corr)) if trial_corr.size else float("nan"),
        "median_trial_corr": float(np.median(trial_corr)) if trial_corr.size else float("nan"),
        "min_trial_corr": float(np.min(trial_corr)) if trial_corr.size else float("nan"),
        "mean_per_channel_corr": float(np.nanmean(ch_corr)),
        "median_per_channel_corr": float(np.nanmedian(ch_corr)),
        "mae": mae,
        "rmse": rmse,
        "relative_rmse": rel_rmse,
    }


# ============================ 通道查找 + MI 可分性 ============================

def best_session_assignment(ours_fps: Sequence[Tuple[float, float]],
                            off_fps: Sequence[Tuple[float, float]],
                            identity_margin: float = 0.5) -> Dict[str, Any]:
    """在一个被试内，把「我们的各 session」与「官方各 session」按幅值指纹做最优配对。

    官方 derivatives 的 session 排序对部分被试与 BIDS sourcedata 不一致（实测）。用
    (std, max_abs) 作为对 ICA 清理稳健的指纹：cost = Σ |log(std比)| + |log(max比)|。
    枚举所有排列取最小 cost。仅当最优排列显著优于「同序(identity)」（差距 > identity_margin）
    才判为「被试 session 顺序被官方重排」，否则保守保持同序，避免幅值相近时的误配。

    返回 {perm（官方索引列表，与我们 session 顺序对齐）, cost, identity_cost, best_cost,
          is_permuted}。
    """
    import itertools

    k = len(ours_fps)

    def _safe_log_ratio(a: float, b: float) -> float:
        a = max(float(a), 1e-9)
        b = max(float(b), 1e-9)
        return abs(math.log(a / b))

    def cost(perm: Tuple[int, ...]) -> float:
        return float(sum(
            _safe_log_ratio(ours_fps[i][0], off_fps[j][0])
            + _safe_log_ratio(ours_fps[i][1], off_fps[j][1])
            for i, j in enumerate(perm)))

    identity = tuple(range(k))
    id_cost = cost(identity)
    best, bc = identity, id_cost
    for perm in itertools.permutations(range(k)):
        c = cost(perm)
        if c < bc - 1e-12:
            bc, best = c, perm
    permuted = (best != identity) and ((id_cost - bc) > identity_margin)
    used = best if permuted else identity
    return {
        "perm": list(used),
        "cost": round(cost(used), 4),
        "identity_cost": round(id_cost, 4),
        "best_cost": round(bc, 4),
        "is_permuted": bool(permuted),
    }


def find_channel_index(channel_names: Sequence[str], target: str) -> Optional[int]:
    """大小写不敏感地查找通道下标；找不到返回 None。"""
    tl = target.strip().lower()
    for i, c in enumerate(channel_names):
        if str(c).strip().lower() == tl:
            return i
    return None


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """两组样本的 Cohen's d（用合并标准差）。退化情形返回 nan。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    if pooled <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / math.sqrt(pooled))


def fisher_score(a: np.ndarray, b: np.ndarray) -> float:
    """两类 Fisher 判别分数 (mean0-mean1)^2 / (var0+var1)。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return float("nan")
    denom = a.var(ddof=1) + b.var(ddof=1)
    if denom <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) ** 2 / denom)


def per_trial_bandpower(freqs: np.ndarray, psd_trials: np.ndarray,
                        band: Tuple[float, float]) -> np.ndarray:
    """每 trial 每通道的频带功率：输入 psd_trials [trials,ch,nfreq]，返回 [trials,ch]。"""
    return bandpower(freqs, psd_trials, band)


def class_band_separability(per_trial_bp_channel: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """对某通道某频带的「每 trial 功率」按左右手两类计算可分性。

    per_trial_bp_channel: [trials]（已选定通道与频带）。
    返回 mean_class0/1、diff、log-power 上的 cohen_d / fisher（功率近似对数正态，取 log 更稳）。
    """
    y = np.asarray(y).ravel()
    p = np.asarray(per_trial_bp_channel, dtype=np.float64)
    c0 = p[y == 0]
    c1 = p[y == 1]
    logp = np.log(np.clip(p, 1e-20, None))
    return {
        "mean_class0": float(c0.mean()) if c0.size else float("nan"),
        "mean_class1": float(c1.mean()) if c1.size else float("nan"),
        "class_diff": float(c0.mean() - c1.mean()) if (c0.size and c1.size) else float("nan"),
        "cohens_d": cohens_d(logp[y == 0], logp[y == 1]),
        "fisher": fisher_score(logp[y == 0], logp[y == 1]),
    }


# ============================ 单 session 一站式质量计算 ============================

def compute_session_quality(
    X_ours: np.ndarray,
    y_ours: np.ndarray,
    X_off: Optional[np.ndarray],
    y_off: Optional[np.ndarray],
    channel_names: Sequence[str],
    sfreq: float,
    *,
    mi_channels: Sequence[str] = MI_CHANNELS_DEFAULT,
    nperseg: int = 256,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """计算一个 session 的全部质量指标。

    返回 (metrics, extras)：
      - metrics：可写进 CSV 的扁平标量字典（ours_* / official_* / *_ratio / MI 可分性等）。
      - extras：给画图用的数组（freqs、PSD、逐通道 ratio、配对相似度明细等）。

    若 X_off/y_off 为 None（官方 .mat 缺失），只计算 ours 侧 + 标记 official_found=False。
    """
    channel_names = [str(c) for c in channel_names]
    metrics: Dict[str, Any] = {}
    extras: Dict[str, Any] = {"channel_names": channel_names}

    # ---- 幅值（ours） ----
    amp_o = amplitude_metrics(X_ours)
    for k, v in amp_o.items():
        metrics[f"ours_{k}"] = v

    # ---- 频域（ours） ----
    f_o, psd_o = welch_psd(X_ours, sfreq, nperseg=nperseg)   # psd_o [trials,ch,nfreq]
    psd_o_chmean = psd_o.mean(axis=(0, 1))                   # [nfreq]
    bp_o = band_powers(f_o, psd_o_chmean)
    for k, v in bp_o.items():
        metrics[f"ours_bp_{k}"] = v

    # ---- 通道级（ours） ----
    rms_o = per_channel_rms(X_ours)
    std_o = per_channel_std(X_ours)

    extras["freqs"] = f_o
    extras["psd_ours_chmean"] = psd_o_chmean
    extras["per_channel_rms_ours"] = rms_o

    has_official = X_off is not None and y_off is not None
    metrics["official_found"] = bool(has_official)

    # ---- MI 可分性（ours），按通道 × {mu_alpha, beta} ----
    # 预先算好每频带的「逐 trial 逐通道」功率，避免每个通道重复算 PSD 积分。
    bp_tc_ours = {b: per_trial_bandpower(f_o, psd_o, EEG_BANDS[b]) for b in ("mu_alpha", "beta")}
    for ch in mi_channels:
        idx = find_channel_index(channel_names, ch)
        metrics[f"{ch}_index"] = idx if idx is not None else -1
        if idx is None:
            extras.setdefault("warnings", []).append(f"通道 {ch} 不在 channel_names 中，跳过其 MI 可分性。")
            continue
        for band_name in ("mu_alpha", "beta"):
            sep = class_band_separability(bp_tc_ours[band_name][:, idx], y_ours)
            metrics[f"ours_{ch}_{band_name}_cohend"] = sep["cohens_d"]
            metrics[f"ours_{ch}_{band_name}_fisher"] = sep["fisher"]
            metrics[f"ours_{ch}_{band_name}_classdiff"] = sep["class_diff"]

    # selected-channel PSD（给 C3/C4/Cz overlay）
    sel_ours: Dict[str, np.ndarray] = {}
    for ch in mi_channels:
        idx = find_channel_index(channel_names, ch)
        if idx is not None:
            sel_ours[ch] = psd_o[:, idx, :].mean(axis=0)
    extras["psd_ours_sel"] = sel_ours

    if not has_official:
        extras["per_channel_rms_ratio"] = None
        return metrics, extras

    # ============ 官方侧 + 对比 ============
    amp_f = amplitude_metrics(X_off)
    for k, v in amp_f.items():
        metrics[f"official_{k}"] = v

    f_f, psd_f = welch_psd(X_off, sfreq, nperseg=nperseg)
    psd_f_chmean = psd_f.mean(axis=(0, 1))
    bp_f = band_powers(f_f, psd_f_chmean)
    for k, v in bp_f.items():
        metrics[f"official_bp_{k}"] = v

    rms_f = per_channel_rms(X_off)

    # ratios
    metrics["std_ratio"] = float(metrics["ours_std"] / metrics["official_std"]) \
        if metrics["official_std"] else float("nan")
    metrics["rms_ratio"] = float(metrics["ours_rms"] / metrics["official_rms"]) \
        if metrics["official_rms"] else float("nan")
    for band_name in list(EEG_BANDS) + ["line_noise_49_51"]:
        o = metrics.get(f"ours_bp_{band_name}", float("nan"))
        ff = metrics.get(f"official_bp_{band_name}", float("nan"))
        metrics[f"bp_ratio_{band_name}"] = float(o / ff) if ff else float("nan")
    metrics["mu_bandpower_ratio"] = metrics.get("bp_ratio_mu_alpha", float("nan"))
    metrics["beta_bandpower_ratio"] = metrics.get("bp_ratio_beta", float("nan"))

    # 逐通道 RMS ratio
    rms_ratio = _safe_ratio(rms_o, rms_f)
    extras["per_channel_rms_ratio"] = rms_ratio
    extras["per_channel_rms_official"] = rms_f
    metrics["per_channel_rms_ratio_median"] = float(np.nanmedian(rms_ratio))
    metrics["per_channel_rms_ratio_min"] = float(np.nanmin(rms_ratio))
    metrics["per_channel_rms_ratio_max"] = float(np.nanmax(rms_ratio))

    extras["psd_off_chmean"] = psd_f_chmean
    sel_off: Dict[str, np.ndarray] = {}
    for ch in mi_channels:
        idx = find_channel_index(channel_names, ch)
        if idx is not None:
            sel_off[ch] = psd_f[:, idx, :].mean(axis=0)
    extras["psd_off_sel"] = sel_off

    # 官方侧 MI 可分性
    bp_tc_off = {b: per_trial_bandpower(f_f, psd_f, EEG_BANDS[b]) for b in ("mu_alpha", "beta")}
    for ch in mi_channels:
        idx = find_channel_index(channel_names, ch)
        if idx is None:
            continue
        for band_name in ("mu_alpha", "beta"):
            sep = class_band_separability(bp_tc_off[band_name][:, idx], y_off)
            metrics[f"official_{ch}_{band_name}_cohend"] = sep["cohens_d"]
            metrics[f"official_{ch}_{band_name}_fisher"] = sep["fisher"]
            metrics[f"official_{ch}_{band_name}_classdiff"] = sep["class_diff"]

    # 标签一致性
    lab = compare_labels(y_ours, y_off)
    metrics.update(lab)

    # 配对相似度（仅形状一致 & exact 时有意义）
    if lab["labels_match_exact"] and X_ours.shape == X_off.shape:
        ps = paired_similarity(X_ours, X_off)
        for k, v in ps.items():
            metrics[f"paired_{k}"] = v
        extras["paired"] = ps
        extras["per_channel_corr"] = per_channel_correlation(X_ours, X_off)
        metrics["paired_done"] = True
    else:
        metrics["paired_done"] = False
        extras["paired"] = None

    return metrics, extras
