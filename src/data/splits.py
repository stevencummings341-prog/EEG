"""按被试(subject-wise)划分工具：41 source / 10 target（骨架）。

铁律（30-model-experiments 规则）：
  - 必须按被试划分，绝不按 trial 划分。
  - target 被试绝不进入 source 训练。
  - 每个划分持久化到 configs/paths.yaml 指定的 splits 目录（JSON），可复现。

当前阶段先留骨架；待预处理跑通后实现。
"""

from __future__ import annotations

from typing import Dict, List


def make_subject_wise_split(
    subjects: List[str],
    n_source: int = 41,
    n_target: int = 10,
    seed: int = 0,
) -> Dict[str, List[str]]:
    """随机生成 subject-wise 划分，返回 {"source": [...], "target": [...]}。

    要求 source 与 target 不相交；n_source + n_target <= len(subjects)。
    TODO: 实现（random.Random(seed) 打乱后切分），并由调用方写入 splits/。
    """
    raise NotImplementedError("make_subject_wise_split 待实现（见 docs/EXPERIMENT_PROTOCOL.md）。")


def save_split(split: Dict[str, List[str]], path) -> None:
    """把划分写入 JSON（供复现）。TODO 实现。"""
    raise NotImplementedError("save_split 待实现。")


def load_split(path) -> Dict[str, List[str]]:
    """读取已保存的划分 JSON。TODO 实现。"""
    raise NotImplementedError("load_split 待实现。")
