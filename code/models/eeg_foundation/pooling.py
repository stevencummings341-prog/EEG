"""Pooling strategies for converting patch sequences to fixed-size representations.

Three strategies:
  1. FlattenPooling: reshape (B, L, D) -> (B, L*D). Preserves all info but high-dim.
  2. AttentionPooling: learnable query attends over patches -> (B, D). Compact.
  3. TemporalBinnedPooling: divide patches into time bins, mean+std -> (B, n_bins*D*2).
     Interpretable for ERP/EEG analysis.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn
import torch.nn.functional as F


class FlattenPooling(nn.Module):
    """Flatten all patches into a single vector.

    Output: (B, patch_num * d_model)
    """

    def __init__(self, d_model: int, patch_num: int):
        super().__init__()
        self.out_dim = d_model * patch_num

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.size(0), -1)


class AttentionPooling(nn.Module):
    """Position-aware attention pooling.

    Uses a learnable query + small scorer network to compute per-patch
    attention weights, then weighted-sums patches to (B, d_model).

    Parameters
    ----------
    d_model : int
        Feature dimension per patch.
    patch_num : int
        Number of patches (for position embedding).
    """

    def __init__(self, d_model: int, patch_num: int):
        super().__init__()
        self.out_dim = d_model
        self.pos_embed = nn.Parameter(torch.randn(patch_num, d_model) * 0.02)
        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.Tanh(),
            nn.Linear(d_model // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, D) -> (B, D)"""
        x = x + self.pos_embed.unsqueeze(0)
        scores = self.scorer(x).squeeze(-1)  # (B, L)
        weights = F.softmax(scores, dim=-1)  # (B, L)
        return (weights.unsqueeze(-1) * x).sum(dim=1)  # (B, D)


class TemporalBinnedPooling(nn.Module):
    """Temporal binned pooling for EEG/ERP signals.

    Divides patches into fixed time bins, computes mean + std per bin.
    Output: (B, n_bins * d_model * 2) if use_std, else (B, n_bins * d_model).

    Default bins for 200Hz, T=170 (850ms window):
      0-150ms, 150-250ms, 250-350ms, 350-450ms, 450-600ms, 600-750ms

    Parameters
    ----------
    d_model : int
        Feature dimension per patch.
    patch_num : int
        Number of patches.
    seq_len : int
        Input sequence length in samples.
    sfreq : float
        Sampling frequency in Hz.
    bin_boundaries_ms : list[float] | None
        Custom bin boundaries. Default uses ERP-relevant windows.
    use_std : bool
        If True, concatenate std to mean (doubles output dim).
    """

    def __init__(
        self,
        d_model: int = 128,
        patch_num: int = 72,
        seq_len: int = 170,
        sfreq: float = 200.0,
        bin_boundaries_ms: Optional[List[float]] = None,
        use_std: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_num = patch_num
        self.use_std = use_std

        if bin_boundaries_ms is None:
            bin_boundaries_ms = [0, 150, 250, 350, 450, 600, 750]
        self.bin_boundaries_ms = bin_boundaries_ms
        self.n_bins = len(bin_boundaries_ms) - 1

        # Compute patch temporal centers
        # ShallowNetEmbedding: conv kernel=25, pool kernel=4, stride=2
        first_center_ms = (25 / 2) / sfreq * 1000
        patch_stride_ms = 2 / sfreq * 1000
        patch_centers_ms = [first_center_ms + i * patch_stride_ms for i in range(patch_num)]

        bin_assignments = []
        for center in patch_centers_ms:
            assigned = False
            for b in range(self.n_bins):
                if bin_boundaries_ms[b] <= center < bin_boundaries_ms[b + 1]:
                    bin_assignments.append(b)
                    assigned = True
                    break
            if not assigned:
                bin_assignments.append(self.n_bins - 1)

        self.register_buffer("bin_assignments", torch.tensor(bin_assignments, dtype=torch.long))
        self.out_dim = self.n_bins * d_model * (2 if use_std else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, D) -> (B, out_dim)"""
        B, L, D = x.shape
        bins = self.bin_assignments
        pooled = []

        for b in range(self.n_bins):
            mask = (bins == b)
            if mask.any():
                bin_feat = x[:, mask, :]
                pooled.append(bin_feat.mean(dim=1))
                if self.use_std:
                    pooled.append(bin_feat.std(dim=1).clamp(min=1e-8))
            else:
                pooled.append(torch.zeros(B, D, device=x.device))
                if self.use_std:
                    pooled.append(torch.zeros(B, D, device=x.device))

        return torch.cat(pooled, dim=-1)
