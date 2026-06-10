"""DeepConvNet baseline (Schirrmeister et al. 2017), PyTorch.

Reference / structure source
----------------------------
Schirrmeister, R. T., et al. (2017). "Deep learning with convolutional neural
networks for EEG decoding and visualization." Human Brain Mapping, 38(11).
We follow the widely-used braindecode-style configuration:

  Block 1 (split temporal + spatial):
    Conv2d(1, F1, (1, k))                # temporal conv (no padding)
    Conv2d(F1, F1, (C, 1))               # spatial conv across all electrodes
    BatchNorm2d(F1) -> ELU -> MaxPool2d((1, p), stride (1, p)) -> Dropout
  Blocks 2-4 (channels F1 -> 50 -> 100 -> 200):
    Conv2d -> BatchNorm2d -> ELU -> MaxPool2d((1, p)) -> Dropout
  Head:
    Flatten -> Linear(flat_dim, n_classes)

Defaults: F1=25, temporal kernel k=10, pool/stride p=3, dropout=0.5, ELU.
Convs use 'valid' padding (no pad), like the original; the flatten dimension is
inferred with a dummy forward so kernel/pool arithmetic can never desync.

Input/Output convention (shared by all models, see .cursor/rules/30-model-experiments):
  forward(x): x = [B, C, T] or [B, 1, C, T]  (C=58 electrodes, T=1000 @ 250 Hz)
  returns dict {"logits": [B, n_classes], "features": [B, flat_dim], "confidence": None}.
Labels are {0,1} (0=left, 1=right); this is a plain classifier (no learned confidence).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class DeepConvNetConfig:
    """DeepConvNet hyperparameters (see configs/session_model_compare.yaml model_params)."""

    n_channels: int = 58
    n_times: int = 1000
    n_classes: int = 2
    n_filters_1: int = 25       # block-1 filter count F1 (then 2x,4x,8x)
    kernel_length: int = 10     # temporal conv kernel (samples)
    pool_size: int = 3          # max-pool kernel & stride (samples)
    dropout: float = 0.5


def _as_4d(x: torch.Tensor) -> torch.Tensor:
    """Coerce input to [B, 1, C, T]; accept [B, C, T] or [B, 1, C, T]."""
    if x.dim() == 3:
        return x.unsqueeze(1)
    if x.dim() == 4:
        if x.shape[1] != 1:
            raise ValueError(f"DeepConvNet expects singleton channel dim=1, got {tuple(x.shape)}")
        return x
    raise ValueError(f"DeepConvNet input must be 3D[B,C,T] or 4D[B,1,C,T], got {tuple(x.shape)}")


class DeepConvNet(nn.Module):
    """DeepConvNet feature extractor + linear head with the unified dict contract."""

    def __init__(self, config: DeepConvNetConfig | None = None):
        super().__init__()
        cfg = config or DeepConvNetConfig()
        self.config = cfg
        F1, k, p = cfg.n_filters_1, cfg.kernel_length, cfg.pool_size
        C = cfg.n_channels
        filters = [F1, F1 * 2, F1 * 4, F1 * 8]   # 25, 50, 100, 200

        # Block 1: temporal conv then spatial conv across electrodes.
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, k), bias=False),
            nn.Conv2d(F1, F1, (C, 1), bias=False),
            nn.BatchNorm2d(F1),
            nn.ELU(),
            nn.MaxPool2d((1, p), stride=(1, p)),
            nn.Dropout(cfg.dropout),
        )
        # Blocks 2-4: temporal conv stacks with growing channel width.
        blocks = []
        in_c = F1
        for out_c in filters[1:]:
            blocks.append(nn.Sequential(
                nn.Conv2d(in_c, out_c, (1, k), bias=False),
                nn.BatchNorm2d(out_c),
                nn.ELU(),
                nn.MaxPool2d((1, p), stride=(1, p)),
                nn.Dropout(cfg.dropout),
            ))
            in_c = out_c
        self.blocks = nn.Sequential(*blocks)

        self.feature_dim = self._infer_feature_dim()
        self.classifier = nn.Linear(self.feature_dim, cfg.n_classes)

    def _infer_feature_dim(self) -> int:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.config.n_channels, self.config.n_times)
            feat = self._features(dummy)
        if was_training:
            self.train()
        return int(feat.shape[1])

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_4d(x)
        x = self.block1(x)
        x = self.blocks(x)
        return torch.flatten(x, start_dim=1)

    def forward(self, x: torch.Tensor) -> dict:
        feats = self._features(x)
        logits = self.classifier(feats)
        return {"logits": logits, "features": feats, "confidence": None}
