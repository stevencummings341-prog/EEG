"""CAP-EEGNet：Confidence-aware Prototype EEGNet（本项目主模型，骨架）。

组成（见 docs/MODEL_PLAN.md，对应 30-model-experiments 规则）：
  - EEGNet Encoder：主干，输入 [batch, 1, 58, 1000]。
  - Adapter：轻量适配模块，用于 target 微调与在线适应（backbone 默认冻结）。
  - Classification Head：分类。
  - Prototype Head：global/subject/session 原型，输出到各类原型的距离/margin。
  - Confidence Head：多源置信度（预测熵、prototype margin、一致性、校准），
    不是简单取 softmax 最大值。

当前阶段（data/paths/preprocessing）先不写复杂实现，仅留架构骨架与接口契约。
等预处理与 41/10 split 跑通后再按 docs/MODEL_PLAN.md 实现 forward/loss。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class CAPEEGNetConfig:
    """CAP-EEGNet 结构超参（与 configs/train_cross_subject.yaml 的 model 段对应）。"""

    n_channels: int = 58
    n_times: int = 1000
    n_classes: int = 2
    # EEGNet encoder
    F1: int = 8
    D: int = 2
    F2: int = 16
    kernel_length: int = 64
    dropout: float = 0.25
    # 各 head 开关
    use_adapter: bool = True
    adapter_bottleneck: int = 16
    use_prototype: bool = True
    prototype_levels: List[str] = field(default_factory=lambda: ["global", "subject", "session"])
    prototype_momentum: float = 0.9
    use_confidence: bool = True
    confidence_sources: List[str] = field(
        default_factory=lambda: ["entropy", "prototype_margin", "consistency", "calibration"]
    )


class CAPEEGNet:  # 占位：实现时改为 torch.nn.Module 子类。
    """CAP-EEGNet 主模型骨架。

    约定：forward(x) 其中 x=[batch, channels, time]，内部 reshape 成
    [batch, 1, channels, time]；返回 dict：
      {
        "logits":      [batch, n_classes],
        "features":    [batch, feat_dim],
        "proto_dist":  [batch, n_classes],   # 到各类原型距离（若启用）
        "confidence":  [batch],              # 样本置信度（若启用）
      }
    """

    def __init__(self, config: CAPEEGNetConfig | None = None):
        self.config = config or CAPEEGNetConfig()
        raise NotImplementedError(
            "CAP-EEGNet 待实现（当前阶段只搭架构；见 docs/MODEL_PLAN.md）。"
        )

    def forward(self, x):  # noqa: D401
        raise NotImplementedError


# 各组件的占位骨架，便于后续分别实现与单测。
class EEGNetEncoder:
    """EEGNet 主干编码器（待实现）。"""

    def __init__(self, config: CAPEEGNetConfig):
        raise NotImplementedError("EEGNetEncoder 待实现。")


class Adapter:
    """轻量 adapter（bottleneck），用于微调/在线适应（待实现）。"""

    def __init__(self, dim: int, bottleneck: int = 16):
        raise NotImplementedError("Adapter 待实现。")


class PrototypeHead:
    """原型头：维护 global/subject/session 原型，高置信样本动量更新（待实现）。"""

    def __init__(self, feat_dim: int, n_classes: int, levels, momentum: float = 0.9):
        raise NotImplementedError("PrototypeHead 待实现。")


class ConfidenceHead:
    """多源置信度头（待实现）。"""

    def __init__(self, feat_dim: int, sources):
        raise NotImplementedError("ConfidenceHead 待实现。")
