"""WBCIC-SHU 2025 三分类（3C）入口：官方 derivatives `.mat` → 统一 npz 约定。

依赖: numpy, scipy

数据事实（已核实 2026-08-04）:
  - 路径: ``<raw>/derivatives/3C dataset_processeddata/sub-XXX/ses-YY/eeg/*.mat``
  - 键: ``data`` float32 **[C, T, N]** = [58, 1000, 300]；``labels`` {1,2,3} 各 100
  - 11 被试 × 3 session = 33 session；与 2C 的 sub-001.. 是**不同被试**（同名勿混）
  - 官方已切段/预处理；本入口**不再**滤波/ICA，只转置 + 标签 {1,2,3}->{0,1,2}

与 2C ``eog_ecg_clean`` 并列：写出独立 processed 树 + manifest，绝不覆盖 2C。
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def load_wbci_3c_session_mat(
    mat_path: str | Path,
    *,
    expect_channels: int = 58,
    expect_times: int = 1000,
    expect_trials: int | None = None,
    # Official sessions are nominally 300 trials; one session has 299 — do not hard-fail.
    min_trials: int = 290,
    expect_n_classes: int = 3,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """读取单个 3C session .mat，返回 (X [n,C,T] float32, y [n] int64{0,1,2}, meta)。"""
    import scipy.io as sio

    mat_path = Path(mat_path)
    m = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    if "data" not in m or "labels" not in m:
        raise KeyError(
            f"{mat_path} 缺少 data/labels 键: {[k for k in m if not str(k).startswith('__')]}"
        )

    raw = np.asarray(m["data"], dtype=np.float32)
    raw_labels = np.ravel(np.asarray(m["labels"])).astype(np.int64)

    # Official layout is [C, T, N]; accept already [N, C, T] if someone re-saves.
    if raw.ndim != 3:
        raise ValueError(f"{mat_path} data ndim {raw.ndim} != 3")
    if raw.shape[0] == expect_channels and raw.shape[1] == expect_times:
        X = np.transpose(raw, (2, 0, 1))  # [N, C, T]
    elif raw.shape[1] == expect_channels and raw.shape[2] == expect_times:
        X = raw
    else:
        raise ValueError(
            f"{mat_path} unexpected data shape {raw.shape}; "
            f"want [C,T,N]=[{expect_channels},{expect_times},N] or [N,C,T]"
        )

    uniq = set(int(v) for v in np.unique(raw_labels).tolist())
    if uniq <= {0, 1, 2}:
        y = raw_labels.astype(np.int64)
    elif uniq <= {1, 2, 3}:
        y = (raw_labels - 1).astype(np.int64)
    else:
        raise ValueError(f"{mat_path} 标签不是 3 类 {{1,2,3}}/{{0,1,2}}: {uniq}")

    n_trials = int(X.shape[0])
    label_counts = {int(k): int(v) for k, v in Counter(y.tolist()).items()}

    status = "ok"
    fail_reasons: List[str] = []
    if X.shape[1] != expect_channels:
        fail_reasons.append(f"n_channels {X.shape[1]} != {expect_channels}")
    if X.shape[2] != expect_times:
        fail_reasons.append(f"n_times {X.shape[2]} != {expect_times}")
    if n_trials != len(y):
        fail_reasons.append(f"n_trials {n_trials} != n_labels {len(y)}")
    if expect_trials is not None and n_trials != expect_trials:
        fail_reasons.append(f"n_trials {n_trials} != {expect_trials}")
    elif n_trials < min_trials:
        fail_reasons.append(f"n_trials {n_trials} < min_trials {min_trials}")
    if len(label_counts) != expect_n_classes:
        fail_reasons.append(f"n_classes {len(label_counts)} != {expect_n_classes}: {label_counts}")
    if fail_reasons:
        status = "failed"

    meta: Dict[str, Any] = {
        "source_mat": str(mat_path),
        "n_trials": n_trials,
        "n_channels": int(X.shape[1]),
        "n_times": int(X.shape[2]),
        "label_counts": label_counts,
        "label_0_count": label_counts.get(0, 0),
        "label_1_count": label_counts.get(1, 0),
        "label_2_count": label_counts.get(2, 0),
        "n_classes": expect_n_classes,
        "units": "uV",
        "entry": "wbci_3c_official_mat",
        "paradigm": "3C",  # left / right / foot-hooking
        "status": status,
        "output_shape": list(X.shape),
    }
    if fail_reasons:
        meta["fail_reasons"] = fail_reasons
    return X, y, meta
