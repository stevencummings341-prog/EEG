"""FBCNet baseline (Mane et al. 2021), PyTorch.

Reference / structure source
----------------------------
Mane, R., et al. (2021). "FBCNet: A Multi-view Convolutional Neural Network for
Brain-Computer Interface." arXiv:2104.01233. Pipeline:

  1. Filter bank: split the signal into ``n_bands`` band-pass views.
  2. Spatial Convolution Block (SCB): a depthwise spatial conv (groups = n_bands),
     so each band learns ``m`` spatial filters over the 58 electrodes; + BatchNorm.
  3. Temporal Variance Layer: split time into ``temporal_segments`` windows and take
     log-variance in each window (variance = the band-power-like nonlinearity).
  4. Linear classifier over the flattened log-variance features.

Implementation assumptions (documented so nothing is hidden):
  * The filter bank is a FIXED (non-trainable) FIR bank: ``n_bands`` band-pass
    kernels designed with ``scipy.signal.firwin`` (Hamming window, odd taps,
    ``pass_zero=False``), applied as a frozen Conv2d over the time axis. This is
    faithful to FBCNet (the filter bank is fixed, only SCB + classifier learn) and
    is differentiable w.r.t. the input and runs on GPU. Default bands = 9 contiguous
    4 Hz bands from 4-40 Hz (4-8, 8-12, ..., 36-40) at 250 Hz.
  * We OMIT the max-norm weight constraints from the original (kept simple for the
    baseline comparison). m=32 spatial filters/band, temporal_segments=4 by default.

Input/Output convention (shared by all models):
  forward(x): x = [B, C, T] or [B, 1, C, T]  (C=58, T=1000 @ 250 Hz)
  returns dict {"logits": [B, n_classes], "features": [B, F*segments], "confidence": None}.
Labels are {0,1}; plain classifier (no learned confidence head).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import firwin


@dataclass
class FBCNetConfig:
    """FBCNet hyperparameters (see configs/session_model_compare.yaml model_params)."""

    n_channels: int = 58
    n_times: int = 1000
    n_classes: int = 2
    sfreq: int = 250
    n_bands: int = 9            # number of band-pass views
    band_width: float = 4.0     # Hz per band
    f_low: float = 4.0          # first band starts here (Hz)
    f_high: float = 40.0        # last band ends here (Hz)
    m: int = 32                 # spatial filters per band (SCB)
    temporal_segments: int = 4  # variance windows over time
    filter_taps: int = 125      # FIR length (odd); ~0.5 s at 250 Hz
    dropout: float = 0.5


def design_filter_bank(cfg: FBCNetConfig) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """Design ``n_bands`` band-pass FIR kernels. Returns (kernels[n_bands, taps], bands)."""
    taps = cfg.filter_taps if cfg.filter_taps % 2 == 1 else cfg.filter_taps + 1
    nyq = cfg.sfreq / 2.0
    bands: List[Tuple[float, float]] = []
    kernels = []
    for i in range(cfg.n_bands):
        lo = cfg.f_low + i * cfg.band_width
        hi = min(cfg.f_low + (i + 1) * cfg.band_width, cfg.f_high, nyq - 1.0)
        if hi <= lo:
            raise ValueError(f"FBCNet band {i} invalid: ({lo},{hi}) Hz at fs={cfg.sfreq}.")
        # FIR band-pass (Hamming-windowed); pass_zero=False => band-pass.
        k = firwin(taps, [lo, hi], pass_zero=False, fs=cfg.sfreq)
        kernels.append(k.astype(np.float32))
        bands.append((lo, hi))
    return np.stack(kernels, axis=0), bands


class FixedFilterBank(nn.Module):
    """Frozen FIR filter bank: [B,1,C,T] -> [B, n_bands, C, T] (same-length conv)."""

    def __init__(self, cfg: FBCNetConfig):
        super().__init__()
        kernels, self.bands = design_filter_bank(cfg)   # [n_bands, taps]
        taps = kernels.shape[1]
        self.pad = taps // 2
        # weight shape [out_channels=n_bands, in_channels=1, 1, taps]
        w = torch.from_numpy(kernels).view(cfg.n_bands, 1, 1, taps)
        self.conv = nn.Conv2d(1, cfg.n_bands, (1, taps), padding=(0, self.pad), bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(w)
        self.conv.weight.requires_grad_(False)   # filter bank is FIXED

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,C,T]; kernel height=1 keeps the electrode axis -> [B, n_bands, C, T]
        return self.conv(x)


class LogVarLayer(nn.Module):
    """Log-variance over ``n_segments`` equal temporal windows.

    Input [B, F, 1, T] -> output [B, F, 1, n_segments]. Variance is the FBCNet
    band-power-like nonlinearity; log stabilizes the scale.
    """

    def __init__(self, n_segments: int):
        super().__init__()
        self.n_segments = int(n_segments)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, f, h, t = x.shape
        seg = t // self.n_segments
        if seg == 0:
            raise ValueError(f"LogVarLayer: T={t} < n_segments={self.n_segments}.")
        x = x[..., : seg * self.n_segments]
        x = x.reshape(b, f, h, self.n_segments, seg)
        v = x.var(dim=-1, unbiased=False)              # [B, F, 1, n_segments]
        return torch.log(torch.clamp(v, min=1e-6))


def _as_4d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x.unsqueeze(1)
    if x.dim() == 4:
        if x.shape[1] != 1:
            raise ValueError(f"FBCNet expects singleton channel dim=1, got {tuple(x.shape)}")
        return x
    raise ValueError(f"FBCNet input must be 3D[B,C,T] or 4D[B,1,C,T], got {tuple(x.shape)}")


class FBCNet(nn.Module):
    """FBCNet: fixed filter bank -> SCB -> log-variance -> linear, unified dict contract."""

    def __init__(self, config: FBCNetConfig | None = None):
        super().__init__()
        cfg = config or FBCNetConfig()
        self.config = cfg
        self.filter_bank = FixedFilterBank(cfg)

        # SCB: depthwise spatial conv per band (groups=n_bands), m filters each.
        scb_out = cfg.n_bands * cfg.m
        self.scb = nn.Sequential(
            nn.Conv2d(cfg.n_bands, scb_out, (cfg.n_channels, 1),
                      groups=cfg.n_bands, bias=False),     # -> [B, n_bands*m, 1, T]
            nn.BatchNorm2d(scb_out),
        )
        self.var_layer = LogVarLayer(cfg.temporal_segments)
        self.dropout = nn.Dropout(cfg.dropout)
        self.feature_dim = scb_out * cfg.temporal_segments
        self.classifier = nn.Linear(self.feature_dim, cfg.n_classes)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_4d(x)
        x = self.filter_bank(x)         # [B, n_bands, C, T]
        x = self.scb(x)                 # [B, n_bands*m, 1, T]
        x = self.var_layer(x)           # [B, n_bands*m, 1, segments]
        x = torch.flatten(x, start_dim=1)
        return self.dropout(x)

    def forward(self, x: torch.Tensor) -> dict:
        feats = self._features(x)
        logits = self.classifier(feats)
        return {"logits": logits, "features": feats, "confidence": None}
