"""EEGNet 编码器/分类器（骨架，待实现）。

参考 Lawhern et al. 2018，适配本项目输入 [batch, 1, 58, 1000] @ 250Hz。
数据集作者的旧实现（只读，MNE0.22/torch1.10 API）可作交叉核对：
  /share/workspace2/moto_imagination/WBCIC_SHU/code/Deep_learning/
请用现代 PyTorch 重写，不要直接 import 旧代码。

约定（20-model-training 规则）：DataLoader 给的是 [batch, channels, time]，
模型内部自行 reshape 成 [batch, 1, channels, time]。
"""

from __future__ import annotations


class EEGNet:  # 占位：实现时改为 torch.nn.Module 子类。
    """EEGNet backbone + 分类头。TODO: 在 Stage 2 实现。"""

    def __init__(self, n_channels: int = 58, n_times: int = 1000, n_classes: int = 2,
                 F1: int = 8, D: int = 2, F2: int = 16, kernel_length: int = 64,
                 dropout: float = 0.25):
        raise NotImplementedError("EEGNet 待实现（Stage 2，见 docs/MODEL_PLAN.md）。")
