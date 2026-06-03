"""SHU 处理后数据的 Dataset（骨架，待实现）。

实现要点（见 30-model-experiments 规则）：
  - 从 configs/paths.yaml 指定的 processed 目录读取
    <processed_dir>/sub-XXX/ses-YY/{X.npy, y.npy}。
  - __getitem__ 返回 (X[channels, time] float32 tensor, y int)。
  - 按被试/ session 组装训练集；subject-wise 划分见 src/data/splits.py。

数据入口是「我们自己 Python 预处理出的 .npy」，不是 derivatives 的 .mat。
"""

from __future__ import annotations

from pathlib import Path
from typing import List


class SHUTrialDataset:  # 占位：实现时改为 torch.utils.data.Dataset 子类。
    """读取处理后 [trials, channels, time] 数据的 Dataset（待实现）。"""

    def __init__(self, processed_root: str | Path, subjects: List[str],
                 sessions: List[int] | None = None):
        raise NotImplementedError(
            "SHUTrialDataset 待实现（当前阶段只搭架构；见 docs/MODEL_PLAN.md）。"
        )
