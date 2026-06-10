"""Channel alignment utilities for cross-dataset MI-EEG experiments."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _norm(ch: str) -> str:
    return ch.strip().upper().replace("FP", "FP")


def get_common_channels(src_channels: List[str], tgt_channels: List[str]) -> Tuple[List[int], List[int], List[str]]:
    """Return source/target indices for channels present in both lists."""
    tgt_lookup = {_norm(ch): i for i, ch in enumerate(tgt_channels)}
    src_idx, tgt_idx, names = [], [], []
    for i, ch in enumerate(src_channels):
        key = _norm(ch)
        if key in tgt_lookup:
            src_idx.append(i)
            tgt_idx.append(tgt_lookup[key])
            names.append(ch)
    return src_idx, tgt_idx, names


def align_channels(X: np.ndarray, src_channels: List[str], tgt_channels: List[str], strategy: str = "common") -> np.ndarray:
    """Align `X [trials, channels, time]` from source channels to target layout.

    `common` returns only the common source-channel subset. `zero_pad` returns the
    target channel count and fills missing channels with zeros.
    """
    X = np.asarray(X)
    src_idx, tgt_idx, _ = get_common_channels(src_channels, tgt_channels)
    if not src_idx:
        raise ValueError("no common channels between source and target")
    if strategy == "common":
        return X[:, src_idx, :]
    if strategy == "zero_pad":
        out = np.zeros((X.shape[0], len(tgt_channels), X.shape[2]), dtype=X.dtype)
        out[:, tgt_idx, :] = X[:, src_idx, :]
        return out
    raise ValueError(f"unknown channel alignment strategy: {strategy}")
