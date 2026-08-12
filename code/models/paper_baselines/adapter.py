"""Project-contract wrappers for the DSGNet paper's baselines.

Each model here comes from **its own authors' official repository** (see ``README.md`` for
provenance and the exhaustive deviation list). This file only translates them into this
project's contract ``{logits, features, confidence}`` — no architecture is touched.

Models (paper reference -> official repo):
  * ``eegnet_official``  [18] https://github.com/vlawhern/arl-eegmodels   (Keras -> ported)
  * ``eegnex``           [20] https://github.com/chenxiachan/EEGNeX       (Keras -> ported)
  * ``eeg_deformer``     [23] https://github.com/yi-ding-cs/EEG-Deformer  (PyTorch, run as-is)

Not included, on purpose: EEG-Inception [27] has no official public release (only a
third-party reimplementation that states it was never checked by the original authors),
MDGEEG [35] ships an empty placeholder repo, and EEG-DG [38] released code whose entry point
imports a model file that is not in the repository and whose network is hard-wired to exactly
two source domains. Reproducing those would mean inventing code, so they are left out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ._official.EEGDeformer import Deformer
from .eegnet_official_torch import EEGNetOfficial
from .eegnex_official_torch import EEGNeXOfficial

BASELINE_NAMES: List[str] = ["eegnet_official", "eegnex", "eeg_deformer"]

#: Learning rates from each model's own official training script. Used when the experiment
#: config leaves ``train.lr`` empty; a config-level ``lr`` always wins.
DEFAULT_LR = {
    "eegnet_official": 1e-3,   # arl-eegmodels examples: Adam defaults
    "eegnex": 1e-3,            # EEGNeX: keras 'adam' default
    "eeg_deformer": 1e-3,      # EEG-Deformer main_*.py: --lr 1e-3
}


@dataclass
class PaperBaselineConfig:
    """Structural hyperparameters; every default is the upstream author default."""

    # data dims (injected by the experiment layer, never hard-coded)
    n_channels: int = 58
    n_times: int = 1000
    n_classes: int = 2
    sfreq: float = 250.0

    variant: str = "eegnet_official"

    # --- EEGNet [18] (arl-eegmodels defaults) ---
    dropoutRate: float = 0.5
    kernLength: int = 64
    F1: int = 8
    D: int = 2
    F2: int = 16
    norm_rate: float = 0.25

    # --- EEG-Deformer [23] (main_MWL.py defaults: kernel 51, T 64, AT 16, layers 6) ---
    temporal_kernel: int = 51
    num_kernel: int = 64
    deformer_depth: int = 4          # = num_layers - 2, as in models/model_handler.py
    heads: int = 16
    mlp_dim: int = 16
    dim_head: int = 16
    deformer_dropout: float = 0.25

    # EEGNeX [20] exposes no structural hyperparameters upstream.


class PaperBaselineClassifier(nn.Module):
    """One published baseline behind this project's dict contract.

    ``forward(x)``: ``x`` is ``[B, C, T]``, returns
    ``{"logits": [B, n_classes], "features": [B, d], "confidence": None}``.

    ``features`` is each model's own penultimate representation (the flattened tensor its
    classifier head consumes), so it is comparable to what the other models report.
    """

    def __init__(self, config: PaperBaselineConfig | None = None):
        super().__init__()
        cfg = config or PaperBaselineConfig()
        self.config = cfg
        variant = (cfg.variant or "").lower()
        if variant not in BASELINE_NAMES:
            raise ValueError(f"unknown baseline '{cfg.variant}' (use {BASELINE_NAMES})")
        self.variant = variant

        if variant == "eegnet_official":
            self.model = EEGNetOfficial(
                nb_classes=cfg.n_classes, Chans=cfg.n_channels, Samples=cfg.n_times,
                dropoutRate=cfg.dropoutRate, kernLength=cfg.kernLength,
                F1=cfg.F1, D=cfg.D, F2=cfg.F2, norm_rate=cfg.norm_rate,
            )
            self.feature_dim = int(self.model.feature_dim)
        elif variant == "eegnex":
            self.model = EEGNeXOfficial(
                n_outputs=cfg.n_classes, n_features=cfg.n_channels, n_timesteps=cfg.n_times,
            )
            self.feature_dim = int(self.model.feature_dim)
        else:  # eeg_deformer — upstream PyTorch, executed unmodified
            self.model = Deformer(
                num_chan=cfg.n_channels, num_time=cfg.n_times,
                temporal_kernel=cfg.temporal_kernel, num_kernel=cfg.num_kernel,
                num_classes=cfg.n_classes, depth=cfg.deformer_depth, heads=cfg.heads,
                mlp_dim=cfg.mlp_dim, dim_head=cfg.dim_head, dropout=cfg.deformer_dropout,
            )
            self.feature_dim = int(self.model.mlp_head[0].in_features)
            self._deformer_features: Optional[torch.Tensor] = None
            # Upstream returns logits only; capture the head's input for `features` without
            # touching the forward maths.
            self.model.mlp_head[0].register_forward_pre_hook(self._capture_deformer)

    def _capture_deformer(self, module: nn.Module, inputs) -> None:
        self._deformer_features = inputs[0]

    def default_lr(self) -> float:
        return DEFAULT_LR[self.variant]

    def describe(self) -> Dict[str, object]:
        return {
            "name": self.variant,
            "source": "official author repository (see code/models/paper_baselines/README.md)",
            "n_params": sum(p.numel() for p in self.parameters()),
            "feature_dim": self.feature_dim,
        }

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        if self.variant == "eeg_deformer":
            logits = self.model(x)
            features = self._deformer_features
            self._deformer_features = None
            if features is None:
                raise RuntimeError(
                    "eeg_deformer: mlp_head hook captured nothing; upstream head structure "
                    "changed and the feature hook must be revisited."
                )
        else:
            features = self.model.forward_features(x)
            logits = self.model.classifier(features)
        return {"logits": logits, "features": features, "confidence": None}


def build_paper_baseline(variant: str, *, n_channels: int, n_times: int, n_classes: int,
                         sfreq: float = 250.0, params: Optional[Dict] = None
                         ) -> PaperBaselineClassifier:
    cfg = PaperBaselineConfig(
        n_channels=n_channels, n_times=n_times, n_classes=n_classes, sfreq=float(sfreq),
        variant=variant, **(params or {}),
    )
    return PaperBaselineClassifier(cfg)
