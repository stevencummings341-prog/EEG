"""PyTorch port of the **official** EEGNet (``EEGNet`` in arl-eegmodels).

Upstream (vendored verbatim at ``_official/EEGModels_arl_keras.py``):
https://github.com/vlawhern/arl-eegmodels — the authors' own Keras implementation, cited as
[18] in the DSGNet paper. Transcribed 1:1 because Keras cannot run in this PyTorch pipeline.

Layer-by-layer correspondence with ``EEGModels.py::EEGNet`` (L127-L153):

    Keras (Chans, Samples, 1)                     this file ([B, 1, C, T])
    Conv2D(F1, (1, kernLength), same, no bias)    conv1
    BatchNormalization()                          bn1
    DepthwiseConv2D((Chans,1), D, max_norm(1.))   depthwise (+ MaxNormKernel(1.0))
    BatchNormalization() -> ELU                   bn2 -> elu
    AveragePooling2D((1,4)) -> Dropout            pool1 -> drop1
    SeparableConv2D(F2, (1,16), same, no bias)    sep_depthwise + sep_pointwise
    BatchNormalization() -> ELU                   bn3 -> elu
    AveragePooling2D((1,8)) -> Dropout            pool2 -> drop2
    Flatten -> Dense(n, max_norm(norm_rate))      flatten -> classifier

``forward`` returns pre-softmax scores; upstream ends in ``softmax`` + categorical
cross-entropy, which is the same objective as ``cross_entropy`` on these scores.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .keras_compat import constrain_max_norm, glorot_uniform_, pad_same_time


class EEGNetOfficial(nn.Module):
    """``EEGNet(nb_classes, Chans, Samples, ...)`` with upstream defaults.

    Upstream defaults are for 128 Hz data; ``kernLength=64`` is kept because the DSGNet paper
    does not state baseline structural hyperparameters, so the authors' defaults are the only
    non-invented choice. See ``README.md`` §Deviations.
    """

    def __init__(
        self,
        nb_classes: int,
        Chans: int = 64,
        Samples: int = 128,
        dropoutRate: float = 0.5,
        kernLength: int = 64,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        norm_rate: float = 0.25,
    ):
        super().__init__()
        self.kernLength = kernLength
        self.sep_kernel = 16

        self.conv1 = glorot_uniform_(nn.Conv2d(1, F1, (1, kernLength), bias=False))
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = constrain_max_norm(
            glorot_uniform_(nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)), 1.0)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.drop1 = nn.Dropout(dropoutRate)

        # SeparableConv2D = depthwise (1,16) followed by pointwise (1,1) to F2.
        self.sep_depthwise = glorot_uniform_(
            nn.Conv2d(F1 * D, F1 * D, (1, self.sep_kernel), groups=F1 * D, bias=False))
        self.sep_pointwise = glorot_uniform_(nn.Conv2d(F1 * D, F2, (1, 1), bias=False))
        self.bn3 = nn.BatchNorm2d(F2)
        self.drop2 = nn.Dropout(dropoutRate)

        self.feature_dim = self._infer_feature_dim(Chans, Samples)
        self.classifier = constrain_max_norm(
            glorot_uniform_(nn.Linear(self.feature_dim, nb_classes)), norm_rate)

    def _infer_feature_dim(self, Chans: int, Samples: int) -> int:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            feat = self.forward_features(torch.zeros(1, 1, Chans, Samples))
        if was_training:
            self.train()
        return int(feat.shape[1])

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:                       # [B, C, T] -> [B, 1, C, T]
            x = x.unsqueeze(1)
        x = self.bn1(self.conv1(pad_same_time(x, self.kernLength)))
        x = self.drop1(F.avg_pool2d(F.elu(self.bn2(self.depthwise(x))), (1, 4)))
        x = self.sep_pointwise(self.sep_depthwise(pad_same_time(x, self.sep_kernel)))
        x = self.drop2(F.avg_pool2d(F.elu(self.bn3(x)), (1, 8)))
        return torch.flatten(x, start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))
