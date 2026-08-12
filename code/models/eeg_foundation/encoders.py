"""EEG/ERP encoders: ShallowNetEmbedding + S4 or Transformer backbone.

Two encoder variants:
  - S4Encoder: CNN stem + S4 blocks (compact, strong inductive bias)
  - TransformerEncoder: CNN stem + Transformer layers (flexible attention)

Both share the same ShallowNetEmbedding stem and output interface:
  (B, T, C) -> (B, patch_num, d_model)
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .s4_layers import S4Encoder as S4EncoderCore


# ── ShallowNet Embedding ─────────────────────────────────────────────────────

class ShallowNetEmbedding(nn.Module):
    """ShallowNet CNN stem from ERP_Benchmark.

    Conv2d temporal -> Conv2d spatial -> BN -> ELU -> AvgPool -> Dropout -> Projection

    Input: (B, T, C) -> permute to (B, 1, C, T)
    Output: (B, patch_num, d_model) where patch_num = (T - 28) // 2 + 1
    """

    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.shallow_net = nn.Sequential(
            nn.Conv2d(1, d_model, (1, 25), (1, 1)),
            nn.Conv2d(d_model, d_model, (c_in, 1), (1, 1)),
            nn.BatchNorm2d(d_model),
            nn.ELU(),
            nn.AvgPool2d((1, 4), (1, 2)),
            nn.Dropout(dropout),
        )
        self.projection = nn.Conv2d(d_model, d_model, (1, 1), stride=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, patch_num, d_model)"""
        x = x.permute(0, 2, 1).unsqueeze(1)  # (B, 1, C, T)
        x = self.shallow_net(x)
        x = self.projection(x)
        x = x.squeeze(2).permute(0, 2, 1)  # (B, patch_num, d_model)
        return x


# ── Transformer Encoder ──────────────────────────────────────────────────────

class TransformerAttention(nn.Module):
    """Standard multi-head self-attention (no causal mask for EEG)."""

    def __init__(self, attention_dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        scale = 1.0 / (E ** 0.5)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous()


class TransformerAttentionLayer(nn.Module):
    """Multi-head attention layer."""

    def __init__(self, attention, d_model: int, n_heads: int):
        super().__init__()
        d_keys = d_model // n_heads
        d_values = d_model // n_heads
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out = self.inner_attention(queries, keys, values)
        out = out.view(B, L, -1)
        return self.out_projection(out)


class TransformerEncoderLayer(nn.Module):
    """Post-LN Transformer encoder layer.

    attn -> residual+dropout -> norm1 -> FFN(Conv1d) -> residual -> norm2
    """

    def __init__(self, attention, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        new_x = self.attention(x, x, x)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(F.gelu(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class TransformerEncoderCore(nn.Module):
    """Stack of TransformerEncoderLayers + final LayerNorm."""

    def __init__(self, attn_layers, norm_layer):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        for attn_layer in self.attn_layers:
            x = attn_layer(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


# ── Combined Encoders ────────────────────────────────────────────────────────

class S4ERPEncoder(nn.Module):
    """S4 layers only (no embedding).

    Input: (B, patch_num, d_model) from ShallowNetEmbedding
    Output: (B, patch_num, d_model)

    Parameters
    ----------
    d_model : int
        Feature dimension (default 128).
    n_layers : int
        Number of S4 blocks (default 4).
    state_dim : int
        S4 hidden state dimension (default 8).
    d_ff : int | None
        FFN intermediate dimension (default 256).
    dropout : float
        Dropout rate (default 0.1).
    """

    def __init__(self, d_model: int = 128, n_layers: int = 4,
                 state_dim: int = 8, d_ff: Optional[int] = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = S4EncoderCore(d_model, n_layers, state_dim, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, patch_num, d_model) -> (B, patch_num, d_model)"""
        return self.encoder(x)


class TransformerERPEncoder(nn.Module):
    """Transformer layers only (no embedding).

    Input: (B, patch_num, d_model) from ShallowNetEmbedding
    Output: (B, patch_num, d_model)

    Parameters
    ----------
    d_model : int
        Feature dimension (default 128).
    n_layers : int
        Number of Transformer layers (default 6).
    n_heads : int
        Number of attention heads (default 8).
    d_ff : int
        FFN intermediate dimension (default 256).
    dropout : float
        Dropout rate (default 0.1).
    """

    def __init__(self, d_model: int = 128, n_layers: int = 6,
                 n_heads: int = 8, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = TransformerEncoderCore(
            [TransformerEncoderLayer(
                TransformerAttentionLayer(TransformerAttention(dropout), d_model, n_heads),
                d_model, d_ff, dropout,
            ) for _ in range(n_layers)],
            norm_layer=nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, patch_num, d_model) -> (B, patch_num, d_model)"""
        return self.encoder(x)
