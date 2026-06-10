"""随机种子工具：保证实验可复现（见 20-model-training 规则）。"""

from __future__ import annotations

import os
import random


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """设置 python / numpy / torch 的随机种子。

    torch 为可选依赖，未安装时静默跳过，方便在纯数据脚本里调用。
    返回所用 seed，便于记录到 run metadata。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            # 牺牲少量速度换取可复现；如需极致性能可在 config 关闭。
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
