"""SHU 2C 数据集预处理核心逻辑（骨架，待实现）。

实现时严格遵循 docs/PREPROCESSING_SPEC.md 与 10-data-preprocessing 规则：
  - 通道：59 EEG + 1 EOG + 4 ECG（顺序未确认，需用 check_raw_bdf.py 核实）。
  - paper_style：去 ECG/EOG -> Pz 重参考并去 Pz(=58) -> 0.5-40Hz 带通 ->
    50Hz 陷波 -> 取 4s MI 段 -> baseline 校正 -> 降采样到 250Hz。
  - 输出 X=[200,58,1000] float32, y=[200] in {0,1}（触发 1->0, 2->1）。
  - 无法得到该形状必须抛错并记录原因，绝不静默裁剪/补零。

注意：本文件目前是骨架。check_raw_bdf.py（Task 1）不依赖它，可先独立运行。
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def preprocess_one_session(
    data_bdf: str,
    evt_bdf: str,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """对单个 subject/session 做预处理。

    参数
    ----
    data_bdf, evt_bdf : 原始 BDF 路径（只读）。
    config : 来自 configs/preprocess.yaml 的字典。

    返回
    ----
    X : [trials, channels, time] = [200, 58, 1000] float32
    y : [trials] int64，取值 {0,1}
    meta : 记录采样率/通道数/事件数/标签分布/形状/状态等（见 spec）。

    TODO: 在 Task 2 实现。先跑 check_raw_bdf.py 确认事件数、通道角色与切窗时机。
    """
    raise NotImplementedError(
        "preprocess_one_session 尚未实现 —— 见 docs/PREPROCESSING_SPEC.md（Task 2）。"
    )
