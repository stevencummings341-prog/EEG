"""SHU 2022 数据集预处理：以作者提供的 per-session .mat 为正式数据入口。

依赖: numpy >= 1.21, scipy >= 1.7

SHU 数据集事实（已核实 2026-06-11）:
  - 每个 session 一个 `sub-XXX_ses-XX_task_motorimagery_eeg.mat`。
  - `data`  : float32 [n_trials, 32, 1000]（µV，250Hz，4s 切段，作者已带通/陷波/切段）。
  - `labels`: int64  [1, n_trials]，取值 {1,2}（1=left, 2=right）。
  - 同名 `.edf` 为连续信号（与 .mat 同尺度），本入口不使用，仅备查。

设计原则:
  - .mat 已是作者发布的预处理结果，本入口**不再做滤波/重参考/ICA**（避免二次处理与
    数据泄露），只做标签归一化 {1,2}->{0,1} 与形状/类别校验，落盘为统一 .npz。
  - 形状不符或单一类别 -> status="failed"（绝不静默裁剪/补零）。
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def load_shu_session_mat(
    mat_path: str | Path,
    *,
    expect_channels: int = 32,
    expect_times: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """读取单个 SHU session .mat，返回 (X [n,32,1000] float32, y [n] int64{0,1}, meta)。"""
    import scipy.io as sio

    mat_path = Path(mat_path)
    m = sio.loadmat(str(mat_path))
    if "data" not in m or "labels" not in m:
        raise KeyError(f"{mat_path} 缺少 data/labels 键: {[k for k in m if not k.startswith('__')]}")

    X = np.asarray(m["data"], dtype=np.float32)            # [n, ch, time]
    raw_labels = np.ravel(np.asarray(m["labels"])).astype(np.int64)

    # 标签归一化: {1,2}->{0,1}（1=left=0, 2=right=1）；已是 {0,1} 则不变。
    uniq = set(int(v) for v in np.unique(raw_labels).tolist())
    if uniq <= {0, 1}:
        y = raw_labels.astype(np.int64)
    elif uniq <= {1, 2}:
        y = (raw_labels - 1).astype(np.int64)
    else:
        raise ValueError(f"{mat_path} 标签非二分类: {uniq}")

    n_trials = int(X.shape[0])
    label_counts = {int(k): int(v) for k, v in Counter(y.tolist()).items()}

    status = "ok"
    fail_reasons: List[str] = []
    if X.ndim != 3:
        fail_reasons.append(f"data ndim {X.ndim} != 3")
    else:
        if X.shape[1] != expect_channels:
            fail_reasons.append(f"n_channels {X.shape[1]} != {expect_channels}")
        if X.shape[2] != expect_times:
            fail_reasons.append(f"n_times {X.shape[2]} != {expect_times}")
    if n_trials != len(y):
        fail_reasons.append(f"n_trials {n_trials} != n_labels {len(y)}")
    if len(label_counts) < 2:
        fail_reasons.append(f"单一类别: {label_counts}")
    if fail_reasons:
        status = "failed"

    meta: Dict[str, Any] = {
        "source_mat": str(mat_path),
        "n_trials": n_trials,
        "n_channels": int(X.shape[1]) if X.ndim == 3 else None,
        "n_times": int(X.shape[2]) if X.ndim == 3 else None,
        "label_counts": label_counts,
        "label_0_count": label_counts.get(0, 0),
        "label_1_count": label_counts.get(1, 0),
        "units": "uV",
        "entry": "author_provided_mat",
        "status": status,
        "output_shape": list(X.shape),
    }
    if fail_reasons:
        meta["fail_reasons"] = fail_reasons
    return X, y, meta
