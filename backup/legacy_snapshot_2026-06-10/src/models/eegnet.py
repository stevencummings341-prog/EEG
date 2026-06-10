"""EEGNet 编码器 + 分类头（PyTorch 实现）。

参考 Lawhern et al. 2018 (EEGNet)，用现代 PyTorch 重写，适配本项目输入
[batch, 1, 58, 1000] @ 250Hz（数据集作者的旧 MNE0.22/torch1.10 实现只读、不直接 import）。

结构（默认 F1=8, D=2, F2=16, kernel_length=64）：
  Block1: Conv2d(1, F1, (1, K), same) -> BN -> DepthwiseConv2d(F1, F1*D, (C,1)) -> BN
          -> ELU -> AvgPool(1,4) -> Dropout
  Block2: SeparableConv2d(F1*D -> F2, (1,16), same) -> BN -> ELU -> AvgPool(1,8) -> Dropout
  Head  : Flatten -> Linear(F2 * (T//32), n_classes)

约定（30-model-experiments 规则）：DataLoader 给 [batch, channels, time]，模型内部自行
reshape 成 [batch, 1, channels, time]。本文件的 forward 同时接受 3D/4D 输入。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class EEGNetConfig:
    """EEGNet 结构超参（与 configs/train_cross_subject.yaml 的 model.encoder 段对应）。"""

    n_channels: int = 58
    n_times: int = 1000
    n_classes: int = 2
    F1: int = 8
    D: int = 2
    F2: int = 16
    kernel_length: int = 64       # 时间卷积核长（~250Hz 下覆盖 256ms）
    pool1: int = 4
    pool2: int = 8
    dropout: float = 0.25


def _as_4d(x: torch.Tensor) -> torch.Tensor:
    """把输入统一成 [B, 1, C, T]：接受 [B, C, T] 或已是 [B, 1, C, T]。"""
    if x.dim() == 3:                      # [B, C, T] -> [B, 1, C, T]
        return x.unsqueeze(1)
    if x.dim() == 4:
        if x.shape[1] != 1:
            raise ValueError(f"EEGNet 期望通道维(dim=1)为 1，得到 {tuple(x.shape)}")
        return x
    raise ValueError(f"EEGNet 输入需为 3D[B,C,T] 或 4D[B,1,C,T]，得到 {tuple(x.shape)}")


class EEGNetEncoder(nn.Module):
    """EEGNet 主干编码器：输入 [B,1,C,T] -> 特征图，flatten 后维度 = feature_dim。

    不含分类头，便于上层组合（CAP-EEGNet 在其上加分类/原型/置信度头）。
    """

    def __init__(self, config: EEGNetConfig | None = None):
        super().__init__()
        cfg = config or EEGNetConfig()
        self.config = cfg
        F1, D, F2 = cfg.F1, cfg.D, cfg.F2
        C, K = cfg.n_channels, cfg.kernel_length

        # Block 1：时间卷积（same padding 保持时间长度）+ 逐通道(depthwise)空间卷积。
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, K), padding="same", bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise：对每个时间特征图独立学一组空间(跨电极)滤波，(C,1) 把 58 电极压成 1。
            nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, cfg.pool1)),
            nn.Dropout(cfg.dropout),
        )
        # Block 2：Separable conv = depthwise (1,16) + pointwise (1,1) 融合到 F2 通道。
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding="same", groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, cfg.pool2)),
            nn.Dropout(cfg.dropout),
        )

        # 用一次 dummy 前向推断 flatten 维度，避免手算 padding 取整出错。
        self._feature_dim = self._infer_feature_dim()

    def _infer_feature_dim(self) -> int:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.config.n_channels, self.config.n_times)
            feat = self.forward_features(dummy)
        if was_training:
            self.train()
        return int(feat.shape[1])

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """返回 flatten 后的特征 [B, feature_dim]。"""
        x = _as_4d(x)
        x = self.block1(x)
        x = self.block2(x)
        return torch.flatten(x, start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


class EEGNet(nn.Module):
    """EEGNet 编码器 + 线性分类头（plain baseline；CAP-EEGNet 的最小骨架基于它）。

    forward(x) 接受 [B,C,T] 或 [B,1,C,T]，返回 logits [B, n_classes]。
    （tensor 返回的旧契约；统一比较框架请用 EEGNetClassifier 的 dict 契约。）
    """

    def __init__(self, config: EEGNetConfig | None = None):
        super().__init__()
        cfg = config or EEGNetConfig()
        self.config = cfg
        self.encoder = EEGNetEncoder(cfg)
        self.classifier = nn.Linear(self.encoder.feature_dim, cfg.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        return self.classifier(feats)


class EEGNetClassifier(nn.Module):
    """EEGNet 基线模型，统一 dict 契约（供 within/cross 比较框架使用）。

    结构来源：Lawhern et al. 2018 (EEGNet)，编码器见 EEGNetEncoder。这是一个
    **纯分类器基线**，不含置信度/原型等组件。

    forward(x): x 为 [B, C, T] 或 [B, 1, C, T]（C=58, T=1000），返回 dict：
      {
        "logits":     [B, n_classes],   # 分类 logits
        "features":   [B, feature_dim],  # 编码器 flatten 特征
        "confidence": None,              # 基线无学习型置信度（评估时用 softmax 概率算校准）
      }
    """

    def __init__(self, config: EEGNetConfig | None = None):
        super().__init__()
        cfg = config or EEGNetConfig()
        self.config = cfg
        self.encoder = EEGNetEncoder(cfg)
        self.feature_dim = self.encoder.feature_dim
        self.classifier = nn.Linear(self.feature_dim, cfg.n_classes)

    def forward(self, x: torch.Tensor) -> dict:
        feats = self.encoder(x)
        logits = self.classifier(feats)
        return {"logits": logits, "features": feats, "confidence": None}
