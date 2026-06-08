"""SHU 处理后数据的 trial 级 Dataset（从正式 .npz 读取 X/y）。

数据入口是我们自己 Python 预处理出的 .npz（每 session 一个，含
X,y,subject_id,session_id,sfreq,channel_names），不是 derivatives 的 .mat。

  - __getitem__ 返回 (X[channels, time] float32 tensor, y int)。
  - 多个 session 的 npz 用 (file_idx, local_idx) 建全局 trial 索引；__init__ 只读每个
    npz 的 y（np.load 惰性，只取 y 很便宜），X 用小 LRU 缓存按需加载，避免一次性把
    全部 session 读进内存。
  - 构造：from_manifest(processed_manifest.csv[, subjects/sessions]) 读 npz_path 列；
    或 from_npz_paths([...])；按被试划分见 src/data/splits.py。
"""

from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.io import load_session_npz
from ..utils.paths import ses_id, sub_id


class SHUTrialDataset(Dataset):
    """读取处理后 [trials, channels, time] .npz 的 trial 级 Dataset。"""

    def __init__(self, npz_paths: Sequence[str | Path], cache_size: int = 4):
        self.npz_paths: List[Path] = [Path(p) for p in npz_paths]
        if not self.npz_paths:
            raise ValueError("SHUTrialDataset 需要至少一个 npz_path。")
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._cache_size = max(1, int(cache_size))

        self._index: List[Tuple[int, int]] = []          # (file_idx, local_trial_idx)
        self.labels: List[int] = []
        self.file_meta: List[Dict[str, Any]] = []
        for fi, p in enumerate(self.npz_paths):
            if not p.exists():
                raise FileNotFoundError(f"npz 不存在: {p}")
            with np.load(p, allow_pickle=False) as d:     # 惰性：只读 y / 标量
                y = np.asarray(d["y"]).astype(np.int64)
                subj, ses = str(d["subject_id"]), str(d["session_id"])
                n = int(y.shape[0])
            self.file_meta.append({"subject_id": subj, "session_id": ses, "n_trials": n, "path": p})
            for li in range(n):
                self._index.append((fi, li))
                self.labels.append(int(y[li]))

    def __len__(self) -> int:
        return len(self._index)

    def _get_X(self, fi: int) -> np.ndarray:
        if fi in self._cache:
            self._cache.move_to_end(fi)
            return self._cache[fi]
        arr = np.asarray(load_session_npz(self.npz_paths[fi])["X"], dtype=np.float32)
        self._cache[fi] = arr
        self._cache.move_to_end(fi)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return arr

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        fi, li = self._index[idx]
        x = torch.from_numpy(np.ascontiguousarray(self._get_X(fi)[li], dtype=np.float32))
        return x, int(self.labels[idx])

    @classmethod
    def from_npz_paths(cls, npz_paths: Sequence[str | Path], **kw) -> "SHUTrialDataset":
        return cls(npz_paths, **kw)

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        subjects: Optional[Sequence[str | int]] = None,
        sessions: Optional[Sequence[str | int]] = None,
        statuses: Sequence[str] = ("ok",),
        **kw,
    ) -> "SHUTrialDataset":
        """从 processed_manifest.csv 的 npz_path 列构造（可按 subject/session/status 过滤）。"""
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"processed manifest 不存在: {manifest_path}。先运行 scripts/preprocess_all.py 生成。"
            )
        subj_set = {sub_id(s) for s in subjects} if subjects else None
        sess_set = {ses_id(s) for s in sessions} if sessions else None
        status_set = set(statuses) if statuses else None

        paths_list: List[str] = []
        with open(manifest_path, "r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if status_set and r.get("status") not in status_set:
                    continue
                if subj_set and r.get("subject_id") not in subj_set:
                    continue
                if sess_set and r.get("session_id") not in sess_set:
                    continue
                npz = (r.get("npz_path") or "").strip()
                if npz:
                    paths_list.append(npz)
        if not paths_list:
            raise ValueError(
                f"manifest {manifest_path} 过滤后无可用 npz_path（statuses={statuses}）。"
            )
        return cls(paths_list, **kw)
