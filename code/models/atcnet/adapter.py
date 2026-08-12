"""Project-facing wrapper for the official ATCNet port.

ATCNet is the strongest published baseline on the SHUv5 three-class LOSO table of the
DSGNet paper (Acc 0.6834 vs DSGNet 0.6856), so it is the calibration point for our
end-to-end runs. ``atcnet_torch.ATCNetOfficial`` is a 1:1 port of the official Keras
``ATCNet_``; this file only translates it into the project contract
``{logits, features, confidence}`` (see ``.cursor/rules/30-model-experiments``).

Upstream folds its ``L2`` kernel regularizers into the loss, so we expose the same
penalty through the trainer's ``uses_custom_loss`` / ``training_step`` hook instead of
silently dropping it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .atcnet_torch import ATCNetOfficial

#: Official recommended recipe (main_TrainValTest.py L413): Adam lr=0.001, batch 64.
DEFAULT_LR = 1e-3


@dataclass
class ATCNetConfig:
    """ATCNet structural hyperparameters — all values are the upstream defaults.

    Data dims come from the experiment layer, never hard-coded here.
    """

    # data dims (shared by every model in this project)
    n_channels: int = 58
    n_times: int = 1000
    n_classes: int = 2
    sfreq: float = 250.0

    # upstream ATCNet_ signature (models.py L34-L37)
    n_windows: int = 5
    eegn_F1: int = 16
    eegn_D: int = 2
    eegn_kernelSize: int = 64
    eegn_poolSize: int = 7
    eegn_dropout: float = 0.3
    tcn_depth: int = 2
    tcn_kernelSize: int = 4
    tcn_filters: int = 32
    tcn_dropout: float = 0.3
    attention: Optional[str] = "mha"
    fuse: str = "average"

    #: Add the official Keras L2 kernel penalties to the loss (upstream behaviour).
    use_official_l2: bool = True


class ATCNetClassifier(nn.Module):
    """Official ATCNet with this project's dict contract.

    ``forward(x)``: ``x`` is ``[B, C, T]``, returns
    ``{"logits": [B, n_classes], "features": [B, d], "confidence": None}``.

    ``features``: ATCNet has no single penultimate embedding — each sliding window feeds
    its own dense head and the *logits* are averaged. We report the mean of the per-window
    TCN features (the tensors the dense heads consume), which is the closest analogue to
    the other models' penultimate features. Nothing in the forward maths is changed.
    """

    def __init__(self, config: ATCNetConfig | None = None):
        super().__init__()
        cfg = config or ATCNetConfig()
        self.config = cfg

        self.model = ATCNetOfficial(
            n_classes=cfg.n_classes,
            in_chans=cfg.n_channels,
            in_samples=cfg.n_times,
            n_windows=cfg.n_windows,
            eegn_F1=cfg.eegn_F1,
            eegn_D=cfg.eegn_D,
            eegn_kernelSize=cfg.eegn_kernelSize,
            eegn_poolSize=cfg.eegn_poolSize,
            eegn_dropout=cfg.eegn_dropout,
            tcn_depth=cfg.tcn_depth,
            tcn_kernelSize=cfg.tcn_kernelSize,
            tcn_filters=cfg.tcn_filters,
            tcn_dropout=cfg.tcn_dropout,
            attention=cfg.attention,
            fuse=cfg.fuse,
        )
        self.feature_dim = int(
            cfg.tcn_filters * (cfg.n_windows if cfg.fuse == "concat" else 1)
        )
        self.uses_custom_loss = bool(cfg.use_official_l2)

    def default_lr(self) -> float:
        return DEFAULT_LR

    def describe(self) -> Dict[str, object]:
        return {
            "name": "atcnet",
            "source": "official Altaheri/EEG-ATCNet ATCNet_ (Keras) ported to PyTorch",
            "n_params": sum(p.numel() for p in self.parameters()),
            "feature_dim": self.feature_dim,
            "n_windows": int(self.model.n_windows),
            "Tc": int(self.model.Tc),
            "Tw": int(self.model.Tw),
            "official_l2": bool(self.uses_custom_loss),
        }

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        logits = self.model(x)
        window_feats = self.model.last_window_features
        if self.config.fuse == "concat":
            features = torch.cat(window_feats, dim=1)
        else:
            features = (window_feats[0] if len(window_feats) == 1
                        else torch.stack(window_feats, dim=0).mean(dim=0))
        return {"logits": logits, "features": features, "confidence": None}

    def training_step(self, x: torch.Tensor, y: torch.Tensor, epoch: int = 0
                      ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """CE + the official Keras L2 kernel penalties (Keras adds them to the loss)."""
        ce = F.cross_entropy(self.forward(x)["logits"], y)
        l2 = self.model.l2_penalty()
        return ce + l2, {"ce": float(ce.detach().item()),
                         "l2": float(l2.detach().item())}
