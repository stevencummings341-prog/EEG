"""Model registry: build any of the four comparison models by name.

All models share the SAME forward contract so the within/cross comparison is fair:
``forward(x)`` accepts ``[B, C, T]`` or ``[B, 1, C, T]`` (C=58, T=1000 @ 250 Hz) and
returns a dict ``{"logits": [B, n_classes], "features": [B, d], "confidence": [B] | None}``.

  * ``eegnet``      -> EEGNetClassifier   (Lawhern 2018) — baseline
  * ``deepconvnet`` -> DeepConvNet        (Schirrmeister 2017) — baseline
  * ``fbcnet``      -> FBCNet             (Mane 2021) — baseline
  * ``cap_eegnet``  -> CAPEEGNet (v1)     — our model (encoder + classifier + learned
                                            confidence head; prototype/online = future)

End-to-end foundation models (advisor-supplied package, see
``code/models/eeg_foundation/README.md``). Same dict contract; the DualCD ones also
expose ``uses_custom_loss`` / ``training_step`` / ``after_optimizer_step`` hooks that
only ``code/training/e2e_trainer.py`` uses:

  * ``s4erp``               -> S4 backbone, flatten pooling, supervised (no DualCD)
  * ``dualcd_s4_pos``       -> S4 + attention pooling + DINO/DualCD
  * ``dualcd_s4_timepatch`` -> S4 + temporal-binned pooling + DINO/DualCD
  * ``dualcd_s4_flatten``   -> S4 + flatten pooling + DINO/DualCD
  * ``dualcd_transformer``  -> Transformer + flatten pooling + DINO/DualCD

Published baselines reproduced for the DSGNet SHUv5 3C comparison. Same dict contract; each
one comes from **its own authors' official repository** (never a third-party reimplementation)
— see ``code/models/atcnet/README.md`` and ``code/models/paper_baselines/README.md``:

  * ``atcnet``          -> ATCNet (Altaheri 2023)  [24] — strongest baseline in the paper
  * ``eegnet_official`` -> EEGNet (Lawhern 2018)   [18] — arl-eegmodels, Keras -> ported
  * ``eegnex``          -> EEGNeX (Chen 2024)      [20] — official Keras -> ported
  * ``eeg_deformer``    -> EEG-Deformer (Ding 2024) [23] — official PyTorch, run unmodified

``eegnet`` (below) is this project's own Phase 0-2c EEGNet and is NOT the same code as
``eegnet_official``; keep them apart when comparing against published numbers.

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
from .atcnet.adapter import ATCNetClassifier, ATCNetConfig
from .cap_eegnet import CAPEEGNet, CAPEEGNetConfig
from .deepconvnet import DeepConvNet, DeepConvNetConfig
from .eeg_foundation.adapter import VARIANT_NAMES as FOUNDATION_MODEL_NAMES
from .eeg_foundation.adapter import EEGFoundationConfig, build_eeg_foundation
from .eegnet import EEGNetClassifier, EEGNetConfig
from .fbcnet import FBCNet, FBCNetConfig
from .paper_baselines.adapter import BASELINE_NAMES as PAPER_BASELINE_NAMES
from .paper_baselines.adapter import PaperBaselineConfig, build_paper_baseline

logger = get_logger("models.registry")

BASELINE_MODEL_NAMES: List[str] = ["eegnet", "deepconvnet", "fbcnet", "cap_eegnet"]
PUBLISHED_MODEL_NAMES: List[str] = ["atcnet"] + list(PAPER_BASELINE_NAMES)
MODEL_NAMES: List[str] = (BASELINE_MODEL_NAMES + list(FOUNDATION_MODEL_NAMES)
                          + PUBLISHED_MODEL_NAMES)


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
    if key == "atcnet":
        cfg = ATCNetConfig(**base, sfreq=float(sfreq), **_select(ATCNetConfig, params))
        return ATCNetClassifier(cfg)
    if key in PAPER_BASELINE_NAMES:
        # `variant` is the model name itself, so drop it from params if a config passes it.
        p = {k: v for k, v in _select(PaperBaselineConfig, params).items()
             if k not in ("variant", "n_channels", "n_times", "n_classes", "sfreq")}
        return build_paper_baseline(key, **base, sfreq=float(sfreq), params=p)
    if key in FOUNDATION_MODEL_NAMES:
        # These need sfreq (DINO band-pass views + temporal-bin pooling boundaries).
        return build_eeg_foundation(key, **base, sfreq=sfreq,
                                    params=_select(EEGFoundationConfig, params))

    raise ValueError(f"unknown model '{name}'. Known models: {MODEL_NAMES}")
