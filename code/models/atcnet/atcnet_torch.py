"""PyTorch port of the **official** ATCNet (``ATCNet_``) from Altaheri/EEG-ATCNet.

Upstream (vendored verbatim next to this file for line-by-line checking):
``_official_keras/models.py`` (``ATCNet_``, ``Conv_block_``, ``TCN_block_``) and
``_official_keras/attention_models.py`` (``attention_block`` / ``mha_block``),
Apache-2.0, https://github.com/Altaheri/EEG-ATCNet

The official release is TensorFlow/Keras and cannot run inside this project's PyTorch
pipeline, so every layer is transcribed here **1:1**: same order, same shapes, same
kernel/pool sizes, same activations, same dropout positions, same residual wiring, same
Keras initializers (glorot_uniform for Conv2D/Dense, he_uniform for the TCN Conv1D),
same ``max_norm`` kernel constraints, and the same ``L2`` kernel penalties (exposed via
``l2_penalty()`` because Keras folds regularizers into the loss, which in PyTorch has to
be added by the training step).

Two mechanical differences are unavoidable and are the only ones:

1. **Data layout.** Keras is channels-last ``(T, C, 1)``; PyTorch is channels-first
   ``[B, 1, T, C]``. Pure re-indexing, no maths.
2. **"same" padding.** Keras/TF puts the extra pad of an even kernel on the *bottom*;
   ``torch``'s ``padding="same"`` puts it on the *left*. We pad explicitly with the TF
   split (``(k-1)//2`` before, ``k//2`` after) so the time alignment matches upstream.

Correctness check: with the upstream BCI-IV-2a dims (22 channels, 1125 samples,
4 classes) this port has **113,732** parameters — the exact count published in the
official README's results table.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization

# Regularization constants, hard-coded inside upstream ``ATCNet_`` (models.py L58-L60).
DENSE_WEIGHT_DECAY = 0.5
CONV_WEIGHT_DECAY = 0.009
CONV_MAX_NORM = 0.6


class _MaxNormKernel(nn.Module):
    """Keras ``max_norm(m, axis=[0,1,2])`` on a conv kernel.

    Keras kernels are ``(..., in, out)`` and the constraint is taken over every axis but
    the last, i.e. one norm per **output filter**. A torch kernel is ``(out, in, ...)``,
    so the same constraint is a renorm along ``dim=0``.
    """

    def __init__(self, max_norm: float = CONV_MAX_NORM):
        super().__init__()
        self.max_norm = float(max_norm)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return w.renorm(p=2, dim=0, maxnorm=self.max_norm)


def _glorot_uniform_(module: nn.Module) -> None:
    """Keras default initializer for Conv2D / Dense."""
    nn.init.xavier_uniform_(module.weight)
    if getattr(module, "bias", None) is not None:
        nn.init.zeros_(module.bias)


def _he_uniform_(module: nn.Module) -> None:
    """Keras ``kernel_initializer='he_uniform'`` (limit = sqrt(6 / fan_in))."""
    fan_in = nn.init._calculate_correct_fan(module.weight, "fan_in")
    limit = math.sqrt(6.0 / fan_in)
    nn.init.uniform_(module.weight, -limit, limit)
    if getattr(module, "bias", None) is not None:
        nn.init.zeros_(module.bias)


def _pad_same_time(x: torch.Tensor, kernel: int) -> torch.Tensor:
    """TF ``padding='same'`` over the time axis (H) of ``[B, C, T, W]``."""
    total = kernel - 1
    return F.pad(x, (0, 0, total // 2, total - total // 2))


class _ConvBlock(nn.Module):
    """``Conv_block_`` (models.py L150-L191), channels-last -> channels-first.

    Keras: ``(T, C, 1)`` -> Conv2D(F1,(kernLength,1),same) -> BN -> DepthwiseConv2D
    ((1,in_chans), depth_multiplier=D) -> BN -> ELU -> AvgPool((8,1)) -> Dropout ->
    Conv2D(F2,(16,1),same) -> BN -> ELU -> AvgPool((poolSize,1)) -> Dropout.
    """

    def __init__(
        self,
        in_chans: int,
        F1: int = 16,
        D: int = 2,
        kern_length: int = 64,
        pool_size: int = 7,
        dropout: float = 0.3,
        max_norm: float = CONV_MAX_NORM,
    ):
        super().__init__()
        F2 = F1 * D
        self.kern_length = kern_length
        self.second_kernel = 16

        self.conv1 = nn.Conv2d(1, F1, (kern_length, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        # DepthwiseConv2D((1, in_chans), depth_multiplier=D) collapses the electrode axis.
        self.depthwise = nn.Conv2d(F1, F2, (1, in_chans), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F2)
        self.pool1 = nn.AvgPool2d((8, 1))          # hard-coded (8,1) upstream
        self.drop1 = nn.Dropout(dropout)
        self.conv3 = nn.Conv2d(F2, F2, (self.second_kernel, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((pool_size, 1))
        self.drop2 = nn.Dropout(dropout)

        for conv in (self.conv1, self.depthwise, self.conv3):
            _glorot_uniform_(conv)
            register_parametrization(conv, "weight", _MaxNormKernel(max_norm))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, T, C]
        x = self.bn1(self.conv1(_pad_same_time(x, self.kern_length)))
        x = self.drop1(self.pool1(F.elu(self.bn2(self.depthwise(x)))))
        x = self.conv3(_pad_same_time(x, self.second_kernel))
        x = self.drop2(self.pool2(F.elu(self.bn3(x))))
        return x                                    # [B, F2, Tc, 1]

    def conv_kernels(self) -> List[torch.Tensor]:
        return [self.conv1.weight, self.depthwise.weight, self.conv3.weight]


class _KerasMHA(nn.Module):
    """``tf.keras.layers.MultiHeadAttention(key_dim, num_heads, dropout)``.

    Not ``nn.MultiheadAttention``: Keras decouples ``key_dim`` from the embedding size
    (here 2 heads x 8 = 16 internal vs 32 embedding), which torch's module cannot express
    because it forces ``head_dim = embed_dim / num_heads``.
    """

    def __init__(self, embed_dim: int, key_dim: int = 8, num_heads: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        inner = num_heads * key_dim
        self.q_proj = nn.Linear(embed_dim, inner)
        self.k_proj = nn.Linear(embed_dim, inner)
        self.v_proj = nn.Linear(embed_dim, inner)
        self.out_proj = nn.Linear(inner, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        for lin in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            _glorot_uniform_(lin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.num_heads, self.key_dim).transpose(1, 2)

        q, k, v = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.key_dim)
        attn = self.attn_drop(torch.softmax(scores, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, T, self.num_heads * self.key_dim)
        return self.out_proj(out)


class _MHABlock(nn.Module):
    """``mha_block`` (attention_models.py L61-L91): LN -> MHA -> Dropout(0.3) -> residual."""

    def __init__(self, embed_dim: int, key_dim: int = 8, num_heads: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mha = _KerasMHA(embed_dim, key_dim=key_dim, num_heads=num_heads,
                             dropout=dropout)
        self.drop = nn.Dropout(0.3)                 # hard-coded upstream

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.mha(self.norm(x)))


class _TCNBlock(nn.Module):
    """``TCN_block_`` (models.py L246-L302): causal dilated residual TCN.

    Per level: Conv1D -> BN -> act -> Dropout, twice, then a residual add and the
    activation. The first level's skip goes through a 1x1 conv only when
    ``input_dimension != filters`` (with the defaults it does not).
    """

    def __init__(self, input_dimension: int, depth: int = 2, kernel_size: int = 4,
                 filters: int = 32, dropout: float = 0.3,
                 max_norm: float = CONV_MAX_NORM):
        super().__init__()
        self.depth = depth
        self.kernel_size = kernel_size
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dilations: List[int] = []

        for level in range(depth):
            dilation = 2 ** level
            in_ch = input_dimension if level == 0 else filters
            self.convs.append(nn.Conv1d(in_ch, filters, kernel_size, dilation=dilation))
            self.convs.append(nn.Conv1d(filters, filters, kernel_size, dilation=dilation))
            self.norms.append(nn.BatchNorm1d(filters))
            self.norms.append(nn.BatchNorm1d(filters))
            self.dilations.append(dilation)

        self.residual_conv: Optional[nn.Conv1d] = None
        if input_dimension != filters:
            self.residual_conv = nn.Conv1d(input_dimension, filters, 1)

        self.drop = nn.Dropout(dropout)
        for conv in list(self.convs) + ([self.residual_conv] if self.residual_conv else []):
            _he_uniform_(conv)
            register_parametrization(conv, "weight", _MaxNormKernel(max_norm))

    def _causal(self, conv: nn.Conv1d, x: torch.Tensor, dilation: int) -> torch.Tensor:
        return conv(F.pad(x, ((self.kernel_size - 1) * dilation, 0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        out = x
        for level in range(self.depth):
            dilation = self.dilations[level]
            h = self._causal(self.convs[2 * level], out, dilation)
            h = self.drop(F.elu(self.norms[2 * level](h)))
            h = self._causal(self.convs[2 * level + 1], h, dilation)
            h = self.drop(F.elu(self.norms[2 * level + 1](h)))
            skip = (self.residual_conv(out)
                    if (level == 0 and self.residual_conv is not None) else out)
            out = F.elu(h + skip)
        return out

    def conv_kernels(self) -> List[torch.Tensor]:
        kernels = [c.weight for c in self.convs]
        if self.residual_conv is not None:
            kernels.append(self.residual_conv.weight)
        return kernels


class ATCNetOfficial(nn.Module):
    """``ATCNet_`` (models.py L34-L117). Defaults are the upstream defaults.

    ``forward`` returns **pre-softmax scores**. Upstream ends in a softmax and trains with
    ``CategoricalCrossentropy(from_logits=False)``, which is mathematically the same
    objective as cross-entropy on these scores; this project's trainer expects logits.
    """

    def __init__(
        self,
        n_classes: int,
        in_chans: int = 22,
        in_samples: int = 1125,
        n_windows: int = 5,
        eegn_F1: int = 16,
        eegn_D: int = 2,
        eegn_kernelSize: int = 64,
        eegn_poolSize: int = 7,
        eegn_dropout: float = 0.3,
        tcn_depth: int = 2,
        tcn_kernelSize: int = 4,
        tcn_filters: int = 32,
        tcn_dropout: float = 0.3,
        attention: Optional[str] = "mha",
        fuse: str = "average",
    ):
        super().__init__()
        if attention not in (None, "mha"):
            raise ValueError(
                f"attention={attention!r} is not ported; upstream ATCNet_ uses 'mha' "
                "(se/cbam/mhla exist in attention_models.py but are not used by ATCNet)."
            )
        if fuse not in ("average", "concat"):
            raise ValueError(f"fuse must be 'average' or 'concat', got {fuse!r}")

        self.n_windows = n_windows
        self.fuse = fuse
        F2 = eegn_F1 * eegn_D
        self.F2 = F2

        self.conv_block = _ConvBlock(
            in_chans=in_chans, F1=eegn_F1, D=eegn_D, kern_length=eegn_kernelSize,
            pool_size=eegn_poolSize, dropout=eegn_dropout,
        )
        self.Tc = (in_samples // 8) // eegn_poolSize
        self.Tw = self.Tc - n_windows + 1
        if self.Tw < 1:
            raise ValueError(
                f"n_windows={n_windows} needs at least that many time steps after the "
                f"conv block, but Tc={self.Tc} (in_samples={in_samples})."
            )

        self.attention_blocks = nn.ModuleList(
            [_MHABlock(F2) for _ in range(n_windows)] if attention == "mha" else []
        )
        self.tcn_blocks = nn.ModuleList([
            _TCNBlock(input_dimension=F2, depth=tcn_depth, kernel_size=tcn_kernelSize,
                      filters=tcn_filters, dropout=tcn_dropout)
            for _ in range(n_windows)
        ])

        if fuse == "average":
            self.dense = nn.ModuleList(
                [nn.Linear(tcn_filters, n_classes) for _ in range(n_windows)]
            )
        else:
            self.dense = nn.ModuleList(
                [nn.Linear(tcn_filters * n_windows, n_classes)]
            )
        for lin in self.dense:
            _glorot_uniform_(lin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> Keras input (1, C, T) permuted to (T, C, 1)
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.dim() != 3:
            raise ValueError(f"ATCNet expects [B, C, T] or [B, 1, C, T], got {tuple(x.shape)}")
        x = x.permute(0, 2, 1).unsqueeze(1)          # [B, 1, T, C]

        feat = self.conv_block(x).squeeze(-1)        # [B, F2, Tc]
        feat = feat.permute(0, 2, 1)                 # [B, Tc, F2]

        window_feats: List[torch.Tensor] = []
        for i in range(self.n_windows):
            block = feat[:, i:i + self.Tw, :]        # [B, Tw, F2]
            if self.attention_blocks:
                block = self.attention_blocks[i](block)
            block = self.tcn_blocks[i](block.transpose(1, 2))   # [B, filters, Tw]
            window_feats.append(block[:, :, -1])                # last time step
        self.last_window_features = window_feats

        if self.fuse == "average":
            outs = [self.dense[i](f) for i, f in enumerate(window_feats)]
            return (outs[0] if len(outs) == 1
                    else torch.stack(outs, dim=0).mean(dim=0))
        return self.dense[0](torch.cat(window_feats, dim=1))

    def l2_penalty(self) -> torch.Tensor:
        """Keras ``L2`` kernel regularizers, which Keras adds to the loss for you.

        ``L2(l)`` in Keras is ``l * sum(w**2)`` (no 1/2 factor). Conv kernels use
        ``conv_weightDecay=0.009``, the classifier heads ``dense_weightDecay=0.5``.
        """
        conv_kernels = self.conv_block.conv_kernels()
        for tcn in self.tcn_blocks:
            conv_kernels += tcn.conv_kernels()
        conv_term = sum((w ** 2).sum() for w in conv_kernels)
        dense_term = sum((lin.weight ** 2).sum() for lin in self.dense)
        return CONV_WEIGHT_DECAY * conv_term + DENSE_WEIGHT_DECAY * dense_term
