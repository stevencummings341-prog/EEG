"""PyTorch port of the **official** EEGNeX (``EEGNeX_8_32``).

Upstream (vendored verbatim at ``_official/BenchmarkModels_eegnex_keras.py``):
https://github.com/chenxiachan/EEGNeX — the authors' own Keras implementation, cited as [20]
in the DSGNet paper. Transcribed 1:1 because Keras cannot run in this PyTorch pipeline.

Layer-by-layer correspondence with ``BenchmarkModels.py::EEGNeX_8_32`` (L603-L639). Upstream
already uses ``data_format="channels_first"`` with input ``(1, n_features, n_timesteps)``, so
the tensor layout matches this project's ``[B, 1, C, T]`` directly:

    Conv2D(8,  (1,64), same, no bias) -> BN -> ELU
    Conv2D(32, (1,64), same, no bias) -> BN                (no activation upstream)
    DepthwiseConv2D((C,1), mult=2, max_norm(1.)) -> BN -> ELU
    AvgPool2D((1,4), same) -> Dropout(0.5)
    Conv2D(32, (1,16), same, dilation=(1,2), no bias) -> BN  (no activation upstream)
    Conv2D(8,  (1,16), same, dilation=(1,4), no bias) -> BN -> ELU
    AvgPool2D((1,4), same) -> Dropout(0.5)
    Flatten -> Dense(n_outputs, max_norm(0.25))

Two upstream quirks are reproduced deliberately (they are easy to "fix" by accident):
the second and the first dilated conv have **no activation** after their BatchNorm, and the
average pooling uses ``padding='same'`` (ceil, padding excluded from the mean).

``forward`` returns pre-softmax scores; upstream ends in ``softmax`` + categorical
cross-entropy, which is the same objective as ``cross_entropy`` on these scores.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .keras_compat import avgpool_same, constrain_max_norm, glorot_uniform_, pad_same_time


class EEGNeXOfficial(nn.Module):
    """``EEGNeX_8_32(n_timesteps, n_features, n_outputs)`` — no free hyperparameters upstream."""

    def __init__(self, n_outputs: int, n_features: int = 22, n_timesteps: int = 1000):
        super().__init__()
        self.conv1 = glorot_uniform_(nn.Conv2d(1, 8, (1, 64), bias=False))
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = glorot_uniform_(nn.Conv2d(8, 32, (1, 64), bias=False))
        self.bn2 = nn.BatchNorm2d(32)

        self.depthwise = constrain_max_norm(
            glorot_uniform_(nn.Conv2d(32, 64, (n_features, 1), groups=32, bias=False)), 1.0)
        self.bn3 = nn.BatchNorm2d(64)
        self.drop1 = nn.Dropout(0.5)

        self.conv3 = glorot_uniform_(nn.Conv2d(64, 32, (1, 16), dilation=(1, 2), bias=False))
        self.bn4 = nn.BatchNorm2d(32)
        self.conv4 = glorot_uniform_(nn.Conv2d(32, 8, (1, 16), dilation=(1, 4), bias=False))
        self.bn5 = nn.BatchNorm2d(8)
        self.drop2 = nn.Dropout(0.5)

        # Inferred by a dummy forward instead of hand-deriving the two ceil-divisions:
        # 'same' pooling rounds up, so the arithmetic is easy to get subtly wrong.
        self.feature_dim = self._infer_feature_dim(n_features, n_timesteps)
        self.classifier = constrain_max_norm(
            glorot_uniform_(nn.Linear(self.feature_dim, n_outputs)), 0.25)

    def _infer_feature_dim(self, n_features: int, n_timesteps: int) -> int:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            feat = self.forward_features(torch.zeros(1, 1, n_features, n_timesteps))
        if was_training:
            self.train()
        return int(feat.shape[1])

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:                       # [B, C, T] -> [B, 1, C, T]
            x = x.unsqueeze(1)
        x = F.elu(self.bn1(self.conv1(pad_same_time(x, 64))))
        x = self.bn2(self.conv2(pad_same_time(x, 64)))          # upstream: no activation
        x = F.elu(self.bn3(self.depthwise(x)))
        x = self.drop1(avgpool_same(x, 4))
        x = self.bn4(self.conv3(pad_same_time(x, 16, dilation=2)))   # upstream: no activation
        x = F.elu(self.bn5(self.conv4(pad_same_time(x, 16, dilation=4))))
        x = self.drop2(avgpool_same(x, 4))
        return torch.flatten(x, start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))
