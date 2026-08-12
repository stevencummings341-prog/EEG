"""S4 (Structured State-Space) core layers.

Pure PyTorch implementation of S4 with HiPPO-LegS initialization
and FFT-based parallel convolution. No external S4/Mamba dependencies.

Reference:
  Gu et al. "Efficiently Modeling Long Sequences with Structured State Spaces" (ICLR 2022)
  Al-Masud et al. "Benchmarking ECG FMs: A Reality Check" (2026)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def make_hippo_legs_matrix(n: int) -> torch.Tensor:
    """Generate HiPPO-LegS (Legendre Sequence) state matrix.

    Provides an optimal basis for compressing continuous-time signals
    into a finite-dimensional state, enabling long-range memory.

    A[i, j] = -(2i+1)^{1/2} (2j+1)^{1/2}  if i > j
               -(i+1)                         if i = j
               0                              if i < j
    """
    P = np.sqrt(1 + 2 * np.arange(n))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = -np.tril(A, k=-1) - np.diag(np.arange(n) + 1)
    return torch.tensor(A, dtype=torch.float32)


class S4Layer(nn.Module):
    """Single Structured State-Space layer.

    Discretized state-space model:
      h_t = A_bar * h_{t-1} + B_bar * x_t
      y_t = C * h_t + D * x_t

    Uses FFT-based convolution for O(L log L) parallel training.

    Parameters
    ----------
    d_model : int
        Input/output feature dimension.
    state_dim : int
        Hidden state dimension (controls memory capacity). Default: 8.
    dropout : float
        Dropout rate. Default: 0.0.
    """

    def __init__(self, d_model: int, state_dim: int = 8, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.H = d_model

        # HiPPO-LegS initialization for A
        A = make_hippo_legs_matrix(state_dim)
        self.A_log = nn.Parameter(torch.log(-A + 1e-8).clamp(min=0))
        self.A_sign = nn.Parameter(torch.sign(-A + 1e-8))

        # Learnable B, C, D
        self.B = nn.Parameter(torch.randn(d_model, state_dim) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, state_dim) * 0.01)
        self.D = nn.Parameter(torch.ones(d_model))

        # Discretization timestep
        self.log_dt = nn.Parameter(torch.rand(d_model) * math.log(0.1) - 4)

        self.dropout = nn.Dropout(dropout)

    def _get_conv_kernel(self, L: int) -> torch.Tensor:
        """Compute SSM convolution kernel K[0..L-1].

        K[t] = C * A_bar^t * B_bar
        """
        A = -torch.exp(self.A_log) * self.A_sign
        dt = torch.exp(self.log_dt)
        A_diag = torch.diagonal(A)
        A_bar = torch.exp(A_diag.unsqueeze(0) * dt.unsqueeze(1))  # (H, N)
        B_bar = self.B * dt.unsqueeze(1)  # (H, N)

        powers = torch.arange(L, device=A.device).float()
        A_pow = A_bar.unsqueeze(-1) ** powers.unsqueeze(0).unsqueeze(0)  # (H, N, L)
        K = torch.einsum('hn,hnl->hl', self.C * B_bar, A_pow)  # (H, L)
        return K

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L, H)

        Returns
        -------
        (B, L, H)
        """
        B, L, H = x.shape
        K = self._get_conv_kernel(L)

        fft_len = 2 * L
        K_fft = torch.fft.rfft(K, n=fft_len)
        X_fft = torch.fft.rfft(x.transpose(1, 2), n=fft_len)
        Y_fft = K_fft.unsqueeze(0) * X_fft
        y = torch.fft.irfft(Y_fft, n=fft_len)[..., :L].transpose(1, 2)

        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        return self.dropout(y)


class S4Block(nn.Module):
    """S4 block with pre-norm, residual, and optional FFN.

    Structure:
      x -> LayerNorm -> S4 -> Dropout -> + residual
      x -> LayerNorm -> FFN -> Dropout -> + residual (optional)
    """

    def __init__(self, d_model: int, state_dim: int = 8,
                 d_ff: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.s4 = S4Layer(d_model, state_dim, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.has_ffn = d_ff is not None
        if self.has_ffn:
            self.norm2 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_ff, d_model), nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.s4(x)
        x = self.dropout1(x) + residual

        if self.has_ffn:
            residual = x
            x = self.norm2(x)
            x = self.ffn(x) + residual
        return x


class S4Encoder(nn.Module):
    """Stack of S4Blocks with final LayerNorm.

    Drop-in replacement for Transformer Encoder.
    """

    def __init__(self, d_model: int, n_layers: int = 4,
                 state_dim: int = 8, d_ff: Optional[int] = None,
                 dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            S4Block(d_model, state_dim, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
