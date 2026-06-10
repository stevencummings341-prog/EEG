"""Model registry: build any of the four comparison models by name.

All models share the SAME forward contract so the within/cross comparison is fair:
``forward(x)`` accepts ``[B, C, T]`` or ``[B, 1, C, T]`` (C=58, T=1000 @ 250 Hz) and
returns a dict ``{"logits": [B, n_classes], "features": [B, d], "confidence": [B] | None}``.

  * ``eegnet``      -> EEGNetClassifier   (Lawhern 2018) — baseline
  * ``deepconvnet`` -> DeepConvNet        (Schirrmeister 2017) — baseline
  * ``fbcnet``      -> FBCNet             (Mane 2021) — baseline
  * ``cap_eegnet``  -> CAPEEGNet (v1)     — our model (encoder + classifier + learned
                                            confidence head; prototype/online = future)

Structural hyperparameters come from ``configs/session_model_compare.yaml``
``model_params.<name>``; data dims (n_channels/n_times/n_classes/sfreq) are passed
explicitly. Unknown params for a model are ignored with a warning (so one shared
config block can carry per-model extras).
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional

import torch.nn as nn

from ..utils.logging_utils import get_logger
from .cap_eegnet import CAPEEGNet, CAPEEGNetConfig
from .deepconvnet import DeepConvNet, DeepConvNetConfig
from .eegnet import EEGNetClassifier, EEGNetConfig
from .fbcnet import FBCNet, FBCNetConfig

logger = get_logger("models.registry")

MODEL_NAMES: List[str] = ["eegnet", "deepconvnet", "fbcnet", "cap_eegnet"]


def _select(cfg_cls, params: Dict) -> Dict:
    """Keep only params that are valid fields of the dataclass; warn on the rest."""
    valid = {f.name for f in dataclasses.fields(cfg_cls)}
    unknown = sorted(set(params) - valid)
    if unknown:
        logger.warning("build_model: ignoring unknown params for %s: %s",
                       cfg_cls.__name__, unknown)
    return {k: v for k, v in params.items() if k in valid}


def build_model(
    name: str,
    *,
    n_channels: int = 58,
    n_times: int = 1000,
    n_classes: int = 2,
    sfreq: int = 250,
    params: Optional[Dict] = None,
) -> nn.Module:
    """Construct a model by name with shared data dims + per-model structural params."""
    key = (name or "").lower().strip()
    params = dict(params or {})
    base = dict(n_channels=n_channels, n_times=n_times, n_classes=n_classes)

    if key == "eegnet":
        cfg = EEGNetConfig(**base, **_select(EEGNetConfig, params))
        return EEGNetClassifier(cfg)
    if key == "deepconvnet":
        cfg = DeepConvNetConfig(**base, **_select(DeepConvNetConfig, params))
        return DeepConvNet(cfg)
    if key == "fbcnet":
        cfg = FBCNetConfig(**base, sfreq=sfreq, **_select(FBCNetConfig, params))
        return FBCNet(cfg)
    if key == "cap_eegnet":
        cfg = CAPEEGNetConfig(**base, **_select(CAPEEGNetConfig, params))
        return CAPEEGNet(cfg)

    raise ValueError(f"unknown model '{name}'. Known models: {MODEL_NAMES}")
