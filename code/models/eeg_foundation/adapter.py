"""Project-facing wrapper for the 5 EEG foundation models (S4 / DINO-DualCD).

Why this file exists
--------------------
The ported package (`models.py`) speaks its own dialect:

  * input  ``(B, T, C)``            — this project's loaders yield ``[B, C, T]``
  * config ``num_channels/num_classes/seq_len/sampling_rate``
                                    — this project passes ``n_channels/n_times/...``
  * output ``S4ERP`` -> dict with ``{logits, z_inv}``,
           DualCD  -> a bare logits tensor
                                    — this project's contract is
                                      ``{logits, features, confidence}`` (see
                                      `.cursor/rules/30-model-experiments`)
  * training DualCD needs ``compute_loss`` + ``update_ema`` + ``update_prototypes``
                                    — plain CE would silently disable DINO/DualCD

`EEGFoundationClassifier` translates all of that, so every runner/experiment can
treat these models exactly like EEGNet. Two extra optional hooks let the
end-to-end trainer drive the self-supervised variants without knowing about them:

  * ``uses_custom_loss`` / ``training_step(x, y, epoch)`` -> (loss, parts)
  * ``after_optimizer_step(x, y)``  (teacher EMA + prototype bank updates)

Variants (``variant=``), matching the advisor's model table:

  | variant              | backbone    | pooling      | DualCD | ~params (32ch/1000T/2C) |
  |:---------------------|:------------|:-------------|:------:|:-----------------------|
  | ``s4erp``            | S4          | flatten      | no     | ~0.9M                  |
  | ``dualcd_s4_pos``    | S4          | attention    | yes    | ~2.0M                  |
  | ``dualcd_s4_timepatch`` | S4       | temporal bin | yes    | ~3.3M                  |
  | ``dualcd_s4_flatten``| S4          | flatten      | yes    | ~65.8M                 |
  | ``dualcd_transformer``| Transformer| flatten      | yes    | ~66.8M                 |
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import (
    S4ERP,
    UnifiedDINODualCD_S4_Flatten,
    UnifiedDINODualCD_S4_Pos,
    UnifiedDINODualCD_S4_Timepatch,
    UnifiedDINODualCD_Transformer,
)

#: variant name -> (class, is_dualcd, default n_layers)
VARIANTS: Dict[str, Tuple[type, bool, int]] = {
    "s4erp": (S4ERP, False, 4),
    "dualcd_s4_pos": (UnifiedDINODualCD_S4_Pos, True, 4),
    "dualcd_s4_timepatch": (UnifiedDINODualCD_S4_Timepatch, True, 4),
    "dualcd_s4_flatten": (UnifiedDINODualCD_S4_Flatten, True, 4),
    "dualcd_transformer": (UnifiedDINODualCD_Transformer, True, 6),
}

VARIANT_NAMES: List[str] = list(VARIANTS)

#: Learning rates recommended by the source package (DualCD is 10x lower).
DEFAULT_LR = {"s4erp": 1e-3, "_dualcd": 1e-4}


@dataclass
class EEGFoundationConfig:
    """Structural hyperparameters for one foundation-model variant.

    Data dims (``n_channels`` / ``n_times`` / ``n_classes`` / ``sfreq``) are supplied
    by the experiment layer, never hard-coded here. Everything else comes from
    ``code/configs/models/<variant>.yaml`` or the experiment's ``model_params``.
    """

    # data dims (shared by every model in this project)
    n_channels: int = 58
    n_times: int = 1000
    n_classes: int = 2
    sfreq: int = 250

    variant: str = "s4erp"

    # backbone
    d_model: int = 128
    n_layers: Optional[int] = None      # None -> variant default (S4: 4, Transformer: 6)
    state_dim: int = 8                  # S4 only
    n_heads: int = 8                    # Transformer only
    d_ff: int = 256
    dropout: float = 0.1

    # DualCD / DINO
    lambda_intra: float = 0.5
    dino_out_dim: int = 256
    proto_k: int = 5
    teacher_momentum: float = 0.996
    warmup_epochs: int = 5

    # DINO multi-view band-pass views (Hz). MI defaults = mu / beta, which replace the
    # source package's ERP defaults (4-12 / 12-30).
    view_low_band: Tuple[float, float] = (8.0, 13.0)
    view_high_band: Tuple[float, float] = (13.0, 30.0)

    # temporal-binned pooling (dualcd_s4_timepatch only), in ms relative to trial start
    bin_boundaries_ms: Optional[List[float]] = field(default=None)
    use_std: bool = True

    # ---- shim so the ported models see their own attribute names ----
    @property
    def num_channels(self) -> int:
        return self.n_channels

    @property
    def num_classes(self) -> int:
        return self.n_classes

    @property
    def seq_len(self) -> int:
        return self.n_times

    @property
    def sampling_rate(self) -> float:
        return float(self.sfreq)

    def resolved_n_layers(self) -> int:
        if self.n_layers is not None:
            return int(self.n_layers)
        return VARIANTS[self.variant][2]


def _as_time_last(x: torch.Tensor) -> torch.Tensor:
    """``[B, C, T]`` or ``[B, 1, C, T]`` -> ``(B, T, C)`` expected by the ported models."""
    if x.dim() == 4:
        if x.shape[1] != 1:
            raise ValueError(f"expected singleton dim=1, got {tuple(x.shape)}")
        x = x.squeeze(1)
    if x.dim() != 3:
        raise ValueError(f"input must be [B,C,T] or [B,1,C,T], got {tuple(x.shape)}")
    return x.transpose(1, 2).contiguous()


class EEGFoundationClassifier(nn.Module):
    """One of the 5 foundation models behind this project's uniform model contract.

    ``forward(x)`` takes ``[B, C, T]`` (or ``[B, 1, C, T]``) and returns
    ``{"logits": [B, n_classes], "features": [B, feature_dim], "confidence": None}``.
    ``features`` is the pooled representation *before* the causal/spurious mask, i.e.
    the same vector ``encode()`` returns, so it is directly comparable to the
    penultimate features of the EEGNet-family baselines.
    """

    def __init__(self, config: Optional[EEGFoundationConfig] = None):
        super().__init__()
        cfg = config or EEGFoundationConfig()
        variant = (cfg.variant or "").lower().strip()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant '{cfg.variant}'. Known: {VARIANT_NAMES}")
        cfg.variant = variant
        cfg.view_low_band = tuple(cfg.view_low_band)
        cfg.view_high_band = tuple(cfg.view_high_band)
        self.config = cfg

        klass, is_dualcd, _ = VARIANTS[variant]
        self.variant = variant
        self.is_dualcd = is_dualcd
        n_layers = cfg.resolved_n_layers()

        common = dict(d_model=cfg.d_model, n_layers=n_layers, d_ff=cfg.d_ff, dropout=cfg.dropout)
        if variant == "s4erp":
            self.model = klass(cfg, state_dim=cfg.state_dim, **common)
        else:
            dualcd = dict(
                lambda_intra=cfg.lambda_intra, dino_out_dim=cfg.dino_out_dim,
                proto_k=cfg.proto_k, teacher_momentum=cfg.teacher_momentum,
            )
            if variant == "dualcd_transformer":
                self.model = klass(cfg, n_heads=cfg.n_heads, **common, **dualcd)
            elif variant == "dualcd_s4_timepatch":
                self.model = klass(cfg, state_dim=cfg.state_dim, **common, **dualcd,
                                   bin_boundaries_ms=cfg.bin_boundaries_ms, use_std=cfg.use_std)
            else:
                self.model = klass(cfg, state_dim=cfg.state_dim, **common, **dualcd)

        self.feature_dim = int(getattr(self.model, "feature_dim", 0)) or self._infer_feature_dim()

    def _infer_feature_dim(self) -> int:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, self.config.n_channels, self.config.n_times)
            dim = int(self.encode(dummy).shape[1])
        if was_training:
            self.train()
        return dim

    # ------------------------------------------------------------------ #
    # inference contract
    # ------------------------------------------------------------------ #
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled representation ``[B, feature_dim]`` (before the DualCD mask)."""
        return self.model.encode(_as_time_last(x))

    def forward(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        xt = _as_time_last(x)
        if self.is_dualcd:
            feats = self.model.encode(xt)
            z_causal, _ = self.model.mask(feats)
            logits = self.model.classifier(z_causal)
        else:
            out = self.model(xt)
            logits, feats = out["logits"], out["z_inv"]
        return {"logits": logits, "features": feats, "confidence": None}

    # ------------------------------------------------------------------ #
    # training hooks (used by code/training/e2e_trainer.py)
    # ------------------------------------------------------------------ #
    @property
    def uses_custom_loss(self) -> bool:
        """True for the DualCD variants, whose loss is not plain cross-entropy."""
        return self.is_dualcd

    def training_step(self, x: torch.Tensor, y: torch.Tensor, epoch: int = 0
                      ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Full training loss for one batch.

        DualCD: ``dino + ibot + 0.1*dkoleo + 0.5*base + 0.5*perturb + 0.1*proto``
        (exact weights live in the ported ``_BaseDualCDModel.compute_loss``).
        S4ERP: plain cross-entropy.
        """
        if self.is_dualcd:
            loss, parts = self.model.compute_loss(
                _as_time_last(x), y, epoch=int(epoch),
                warmup_epochs=self.config.warmup_epochs,
            )
            return loss, {k: float(v.detach().item()) for k, v in parts.items()}
        logits = self.forward(x)["logits"]
        loss = F.cross_entropy(logits, y)
        return loss, {"ce": float(loss.detach().item())}

    @torch.no_grad()
    def after_optimizer_step(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Teacher EMA + prototype-bank update (DualCD only; no-op otherwise)."""
        if not self.is_dualcd:
            return
        self.model.update_ema()
        self.model.update_prototypes(_as_time_last(x), y)

    def default_lr(self) -> float:
        return DEFAULT_LR["s4erp"] if self.variant == "s4erp" else DEFAULT_LR["_dualcd"]

    def describe(self) -> Dict[str, object]:
        n_all = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "variant": self.variant,
            "is_dualcd": self.is_dualcd,
            "feature_dim": self.feature_dim,
            "n_params": n_all,
            "n_trainable_params": n_trainable,
            "n_channels": self.config.n_channels,
            "n_times": self.config.n_times,
            "n_classes": self.config.n_classes,
            "sfreq": self.config.sfreq,
        }


def build_eeg_foundation(variant: str, *, n_channels: int, n_times: int, n_classes: int,
                         sfreq: int, params: Optional[Dict] = None) -> EEGFoundationClassifier:
    """Build a variant with data dims + structural params (used by the model registry)."""
    import dataclasses

    valid = {f.name for f in dataclasses.fields(EEGFoundationConfig)}
    extra = {k: v for k, v in dict(params or {}).items() if k in valid}
    cfg = EEGFoundationConfig(
        n_channels=n_channels, n_times=n_times, n_classes=n_classes, sfreq=sfreq,
        variant=variant, **{k: v for k, v in extra.items() if k != "variant"},
    )
    return EEGFoundationClassifier(cfg)


def normalize_trials(x: torch.Tensor, mode: str = "per_sample_zscore",
                     eps: float = 1e-8) -> torch.Tensor:
    """Fit-free per-trial normalization for ``[N, C, T]`` (no train/test leakage).

    ``per_sample_zscore``  : z-score each (trial, channel) over time — the source
                             package's recommended default for these models.
    ``per_sample_global``  : z-score each trial over channels+time jointly.
    ``none``               : passthrough.

    All modes use only the trial itself, so applying them to test data leaks nothing.
    """
    mode = (mode or "none").lower()
    if mode in ("none", "", "raw"):
        return x
    if mode == "per_sample_zscore":
        m = x.mean(dim=-1, keepdim=True)
        s = x.std(dim=-1, keepdim=True).clamp_min(eps)
        return (x - m) / s
    if mode == "per_sample_global":
        flat = x.reshape(x.shape[0], -1)
        m = flat.mean(dim=1).reshape(-1, 1, 1)
        s = flat.std(dim=1).clamp_min(eps).reshape(-1, 1, 1)
        return (x - m) / s
    raise ValueError(
        f"unknown normalization '{mode}' "
        "(use per_sample_zscore | per_sample_global | none)."
    )


__all__: Sequence[str] = [
    "EEGFoundationConfig", "EEGFoundationClassifier", "build_eeg_foundation",
    "normalize_trials", "VARIANTS", "VARIANT_NAMES", "DEFAULT_LR",
]
