"""SHU 处理后数据的 Dataset 与按被试划分工具（骨架，待实现）。

实现要点（见 30-experiment-protocol 规则）：
  - SHUTrialDataset：从 data/<variant>/sub-XXX/ses-YY/{X.npy,y.npy} 读取，
    __getitem__ 返回 (X[channels, time] float32 tensor, y int)。
  - make_subject_wise_split：随机选 41 source / 10 target，持久化到
    outputs/splits/<name>.json；保证 target 不泄漏进 source。
"""

from __future__ import annotations

from typing import Dict, List


def make_subject_wise_split(
    subjects: List[str],
    n_source: int = 41,
    n_target: int = 10,
    seed: int = 0,
) -> Dict[str, List[str]]:
    """随机生成 subject-wise 划分。

    返回 {"source": [...], "target": [...]}，二者不相交。
    TODO: 在 Stage 3 实现，并把划分写入 outputs/splits/。
    """
    raise NotImplementedError("make_subject_wise_split 待实现（Stage 3）。")
