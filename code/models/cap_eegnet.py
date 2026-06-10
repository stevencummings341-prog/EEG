"""CAP-EEGNet：本项目主模型，最终对齐学长聊天记录的完整方案。

> 最终目标（见 docs/references/ChatGPT-EEG-MI-pretraining.md sec.13-16 与 docs/ROADMAP.md）：
> **Confidence-aware Online Adaptive Multi-Subagent Pretraining Framework for
> Cross-subject MI EEG Decoding** —— 面向跨被试运动想象 EEG 的「置信度感知 + 在线自适应
> + 多神经子模块」预训练框架。CAP-EEGNet 是它的具体模型实例。

**这不是一个普通 EEGNet 分类器。** full CAP-EEGNet 应包含（全部为可微分深度模块，
不是手工特征拼接）：
  1. (main encoder) EEGNet-style encoder —— 主干，输入 [batch, 1, 58, 1000]。
  2. (neural subagents) 多神经子模块 / neural experts：时频、空间拓扑、熵/复杂度、
     动态连接等视角，各自端到端可微（NOT 手工 CSP/DE/TE 直接拼接），可靠性加权融合。
  3. (confidence head) 置信度预测头：多源融合（预测熵 + prototype margin + 一致性 +
     OOD/校准），**绝不只取 softmax 最大值**。
  4. (prototype memory) 原型头/记忆：global / subject / session 三级原型，高置信样本动量更新。
  5. (adapter) 轻量适配模块：target 微调与在线适应的主要可训练部分（backbone 默认冻结）。
  6. (domain alignment) subject/session 对齐：类别可分、被试不可分。
  7. (online update) 在线更新模块：test-then-update，只更新轻量模块。
  (可选 v2) dataset-aware neural router：按数据集特性给各 subagent 分配权重。

--------------------------------------------------------------------------------
**当前实现状态：CAP-EEGNet v1（cross-session 比较框架用）。**
v1 = EEGNet Encoder + Classification Head + **轻量学习型 Confidence Head**（单源、
端到端学习的标量置信度，用 calibration 风格的辅助损失训练）。它能公平参加 within/cross
两个协议，并提供一个比 softmax-max 更有意义的「学习型」置信度雏形。

**v1 明确还不是论文最终方法**（不要当成完整方法汇报）。下列 full 组件 **尚未实现**，
保留 config flag + 占位类 + 明确 NotImplementedError，按 docs/ROADMAP.md 逐个补齐：
  - neural subagents（多神经子模块）       use_subagents
  - dataset-aware router（v2）              use_dataset_router
  - adapter（轻量适配）                     use_adapter
  - prototype memory（三级原型）            use_prototype
  - **多源** confidence（熵+margin+一致性+OOD，取代 v1 的单源学习头） full ConfidenceHead
  - domain alignment（被试/session 对齐）   use_domain_align
  - online update（test-then-update）        use_online_update
启用任一未实现组件会抛带明确提示的 NotImplementedError。v1 的 confidence 是
「单源学习型标量」，不是上面的「多源融合」——文档中务必标注 v1，避免伪装成完整方法。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .eegnet import EEGNetConfig, EEGNetEncoder

# 统一的「未实现」提示语：明确这是为 full CAP-EEGNet 预留、minimal sanity 模型不含。
_RESERVED = (
    "Reserved for full CAP-EEGNet (Stage 2/3, see docs/ROADMAP.md); "
    "NOT implemented in the minimal sanity model (Stage 0)."
)


@dataclass
class CAPEEGNetConfig:
    """CAP-EEGNet 结构超参（与 configs/train_cross_subject.yaml 的 model 段对应）。

    flags 默认值反映「当前只实现 minimal」：高级组件一律 False；置 True 会触发明确的
    NotImplementedError，避免误以为 full 版已就绪。
    """

    n_channels: int = 58
    n_times: int = 1000
    n_classes: int = 2
    # —— Stage 0 已实现：EEGNet main encoder ——
    F1: int = 8
    D: int = 2
    F2: int = 16
    kernel_length: int = 64
    dropout: float = 0.25
    # —— Stage 2+（尚未实现）：full CAP-EEGNet 各组件开关 ——
    use_subagents: bool = False           # 多神经子模块编码器（neural experts）
    subagent_views: List[str] = field(
        default_factory=lambda: ["temporal_spectral", "spatial_topology",
                                 "entropy_complexity", "connectivity"]
    )
    use_dataset_router: bool = False      # dataset-aware neural router（可选 v2）
    use_adapter: bool = False             # 轻量 adapter（微调/在线）
    adapter_bottleneck: int = 16
    use_prototype: bool = False           # global/subject/session 原型记忆
    prototype_levels: List[str] = field(default_factory=lambda: ["global", "subject", "session"])
    prototype_momentum: float = 0.9
    use_confidence: bool = False          # v1: 轻量学习型置信度头（单源标量，已实现）
    confidence_hidden: int = 32           # v1 confidence MLP 隐藏维
    confidence_weight: float = 0.1        # v1 confidence 校准损失权重（trainer 读取）
    confidence_sources: List[str] = field(
        default_factory=lambda: ["entropy", "prototype_margin", "consistency", "ood_calibration"]
    )  # 多源 confidence（full 版，尚未实现）
    use_domain_align: bool = False        # subject/session 对齐（类可分、被试不可分）
    use_online_update: bool = False       # 在线 test-then-update 更新模块

    def to_encoder_config(self) -> EEGNetConfig:
        return EEGNetConfig(
            n_channels=self.n_channels,
            n_times=self.n_times,
            n_classes=self.n_classes,
            F1=self.F1,
            D=self.D,
            F2=self.F2,
            kernel_length=self.kernel_length,
            dropout=self.dropout,
        )

    def any_full_component_enabled(self) -> List[str]:
        """返回被打开但尚未实现的 full 组件名（用于 fail-fast）。

        注意：use_confidence 不在此列——v1 已实现「轻量学习型」confidence 头。
        （full 的「多源」confidence 仍未实现；见 ConfidenceHead 占位类。）
        """
        enabled = []
        for name, flag in (
            ("subagents", self.use_subagents),
            ("dataset_router", self.use_dataset_router),
            ("adapter", self.use_adapter),
            ("prototype", self.use_prototype),
            ("domain_align", self.use_domain_align),
            ("online_update", self.use_online_update),
        ):
            if flag:
                enabled.append(name)
        return enabled


class CAPEEGNet(nn.Module):
    """CAP-EEGNet 主模型（当前 = v1）。

    **v1** = EEGNet Encoder + 线性分类头 (+ 可选 v1 学习型 Confidence Head)。
    forward(x) 中 x = [batch, channels, time] 或 [batch, 1, channels, time]；返回 dict
    （契约稳定，后续加 head 时只填充 None 字段，不改键）：
      {
        "logits":     [batch, n_classes],   # 已实现
        "features":   [batch, feat_dim],     # 已实现（编码器特征）
        "proto_dist": None,                  # 未实现（future）：Prototype Head/Memory
        "confidence": [batch] 或 None,        # v1 学习型置信度（use_confidence=True 时）
      }

    confidence (v1)：在编码器特征上接一个小 MLP -> sigmoid 标量 ∈ (0,1)，由 trainer 用
    「预测是否正确」作为目标做 calibration 风格 BCE 训练（权重 = confidence_weight）。
    它**不**等于 softmax 最大值，但也**还不是** full 版的多源融合置信度（见模块 docstring）。
    """

    def __init__(self, config: CAPEEGNetConfig | None = None):
        super().__init__()
        self.config = config or CAPEEGNetConfig()

        # full 组件被打开但未实现 -> 立刻报错，杜绝「以为是 full 实则 v1」。
        # （use_confidence 不在 fail-fast 列表里：v1 已实现学习型 confidence。）
        enabled = self.config.any_full_component_enabled()
        if enabled:
            raise NotImplementedError(
                f"CAP-EEGNet full components requested but not implemented: {enabled}. "
                + _RESERVED
            )

        self.encoder = EEGNetEncoder(self.config.to_encoder_config())
        self.feature_dim = self.encoder.feature_dim
        self.classification_head = nn.Linear(self.feature_dim, self.config.n_classes)

        # —— v1 学习型 confidence 头（可选）——
        self.confidence_weight = float(self.config.confidence_weight)
        if self.config.use_confidence:
            self.confidence_head: Optional[nn.Module] = ConfidenceHeadV1(
                self.feature_dim, hidden=self.config.confidence_hidden
            )
        else:
            self.confidence_head = None

        # —— full CAP-EEGNet 预留组件（v1 不构建；占位为 None）——
        self.subagents: Optional[nn.Module] = None        # future: NeuralSubagentEncoder
        self.dataset_router: Optional[nn.Module] = None   # future(v2): DatasetAwareRouter
        self.adapter: Optional[nn.Module] = None          # future: Adapter
        self.prototype_head: Optional[nn.Module] = None   # future: PrototypeMemory
        self.domain_head: Optional[nn.Module] = None      # future: DomainAlignmentHead
        self.online_module: Optional[nn.Module] = None    # future: OnlineUpdateModule

    def forward(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        features = self.encoder(x)               # [B, feat_dim]
        logits = self.classification_head(features)
        confidence = None
        if self.confidence_head is not None:
            confidence = self.confidence_head(features).squeeze(-1)   # [B] in (0,1)
        return {
            "logits": logits,
            "features": features,
            "proto_dist": None,                  # future: Prototype Head/Memory
            "confidence": confidence,            # v1 learned scalar OR None
        }

    # ------------------------------------------------------------------ #
    # 以下为 full CAP-EEGNet 的预留接口（Stage2+ 实现）。
    # ------------------------------------------------------------------ #
    def predict_confidence(self, x: torch.Tensor) -> torch.Tensor:
        """样本置信度。

        v1：若构建了学习型 confidence 头，返回其标量置信度 [B] ∈ (0,1)（端到端学习，
        非 softmax 最大值）。full 版的「多源」置信度（熵 + prototype margin + 一致性 +
        OOD/校准）尚未实现。
        """
        if self.confidence_head is not None:
            return self.forward(x)["confidence"]
        raise NotImplementedError(
            "predict_confidence: multi-source confidence not implemented; "
            "enable use_confidence for the v1 learned head. " + _RESERVED
        )

    def online_update(self, *args, **kwargs):
        """test-then-update 在线更新（只更新 adapter/prototype/calibration/BN）。

        backbone 默认冻结，绝不默认整骨干更新。TODO(Stage4)。
        """
        raise NotImplementedError("online_update: " + _RESERVED)


# --------------------------------------------------------------------------- #
# v1 已实现组件：轻量学习型 confidence 头（单源标量）。
# --------------------------------------------------------------------------- #
class ConfidenceHeadV1(nn.Module):
    """v1 学习型置信度头：encoder 特征 -> 小 MLP -> sigmoid 标量 ∈ (0,1)。

    输入 features [B, feat_dim]，输出 confidence [B, 1]。由 trainer 用「预测是否正确」
    作为目标做 calibration 风格 BCE 训练（detached 特征/目标，不影响分类梯度方向）。

    **这是 v1（单源、学习型）**，刻意不等于 softmax 最大值；但也还不是 full 版的多源
    融合置信度（predictive entropy + prototype margin + consistency + OOD/calibration，
    见下方 ConfidenceHead 占位类）。
    """

    def __init__(self, feat_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)   # [B, 1]


# --------------------------------------------------------------------------- #
# full CAP-EEGNet 各组件的占位骨架（接口契约保留，future 逐个实现与单测）。
# 全部为「可微分神经模块」，不是手工特征提取器（见 docs/MODEL_PLAN.md）。
# --------------------------------------------------------------------------- #
class NeuralSubagentEncoder(nn.Module):
    """多神经子模块编码器（neural experts）：时频/空间拓扑/熵复杂度/连接 等视角，
    各自端到端可微，可靠性加权融合后送主编码器。TODO(Stage2)。"""

    def __init__(self, config: CAPEEGNetConfig, views: List[str] | None = None):
        super().__init__()
        raise NotImplementedError("NeuralSubagentEncoder: " + _RESERVED)


class DatasetAwareRouter(nn.Module):
    """dataset-aware neural router：按数据集元特征/统计/probe 给各 subagent 分配权重。
    可选 v2 模块。TODO。"""

    def __init__(self, n_subagents: int):
        super().__init__()
        raise NotImplementedError("DatasetAwareRouter: " + _RESERVED)


class Adapter(nn.Module):
    """轻量 adapter（bottleneck），target 微调/在线适应的主要可训练部分。TODO(Stage2)。"""

    def __init__(self, dim: int, bottleneck: int = 16):
        super().__init__()
        raise NotImplementedError("Adapter: " + _RESERVED)


class PrototypeMemory(nn.Module):
    """原型记忆：global/subject/session 三级原型，高置信样本动量更新；
    输出 distance_to_own / nearest_wrong / margin。TODO(Stage2)。"""

    def __init__(self, feat_dim: int, n_classes: int, levels, momentum: float = 0.9):
        super().__init__()
        raise NotImplementedError("PrototypeMemory: " + _RESERVED)


class ConfidenceHead(nn.Module):
    """多源置信度头：predictive entropy / prototype margin / consistency / OOD-calibration。
    **不是 softmax 最大值。** TODO(Stage2)。"""

    def __init__(self, feat_dim: int, sources):
        super().__init__()
        raise NotImplementedError("ConfidenceHead: " + _RESERVED)


class DomainAlignmentHead(nn.Module):
    """subject/session 对齐头（adversarial / CORAL / MMD / prototype align）：
    min H(Y|Z) 且 max H(S|Z)。TODO(Stage2)。"""

    def __init__(self, feat_dim: int, n_subjects: int):
        super().__init__()
        raise NotImplementedError("DomainAlignmentHead: " + _RESERVED)


class OnlineUpdateModule(nn.Module):
    """在线更新模块：test-then-update，置信度门控，只更新 adapter/prototype/calib/BN，
    带 EMA teacher / replay / 动量 / 蒸馏 等稳定机制。TODO(Stage4)。"""

    def __init__(self, config: CAPEEGNetConfig):
        super().__init__()
        raise NotImplementedError("OnlineUpdateModule: " + _RESERVED)
