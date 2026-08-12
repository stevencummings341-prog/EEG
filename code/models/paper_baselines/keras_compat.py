"""Keras-to-PyTorch primitives shared by the vendored Keras baselines.

The DSGNet paper's baselines EEGNet [18] and EEGNeX [20] are released as Keras/TensorFlow
code (see ``_official/``). Transcribing them faithfully needs three things PyTorch does not
provide out of the box:

1. **TF ``padding='same'``** — TF puts the extra pad of an even/dilated kernel *after* the
   signal; ``torch``'s ``padding="same"`` puts it *before*. A one-sample time shift is not a
   maths error but it is an avoidable difference, so we pad explicitly with the TF split.
2. **``max_norm`` kernel constraints** — Keras renormalizes a weight tensor after every
   update. We use ``torch.nn.utils.parametrize`` so the constraint is applied on every access
   (equivalent, and it survives ``state_dict`` round-trips).
3. **Keras default initializers** — ``glorot_uniform`` for Conv2D/Dense with zero bias,
   whereas torch defaults to Kaiming-uniform with ``a=sqrt(5)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization


class MaxNormKernel(nn.Module):
    """Keras ``max_norm(m)`` on a kernel: one L2 norm per output filter/unit.

    Keras kernels are ``(..., in, out)`` and torch kernels are ``(out, in, ...)``, so the
    per-output-filter constraint is a renorm along ``dim=0`` here.
    """

    def __init__(self, max_norm: float = 1.0):
        super().__init__()
        self.max_norm = float(max_norm)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return w.renorm(p=2, dim=0, maxnorm=self.max_norm)


def constrain_max_norm(module: nn.Module, max_norm: float) -> nn.Module:
    register_parametrization(module, "weight", MaxNormKernel(max_norm))
    return module


def glorot_uniform_(module: nn.Module) -> nn.Module:
    """Keras default init for Conv2D / Dense (bias zeros)."""
    nn.init.xavier_uniform_(module.weight)
    if getattr(module, "bias", None) is not None:
        nn.init.zeros_(module.bias)
    return module


def pad_same_time(x: torch.Tensor, kernel: int, dilation: int = 1) -> torch.Tensor:
    """TF ``padding='same'`` over the LAST axis of ``[B, C, H, W]`` (time = W).

    Effective kernel with dilation is ``(k-1)*d + 1``; TF splits the total padding as
    ``floor(total/2)`` before and the remainder after.
    """
    total = (kernel - 1) * dilation
    return F.pad(x, (total // 2, total - total // 2, 0, 0))


def avgpool_same(x: torch.Tensor, pool_w: int) -> torch.Tensor:
    """Keras ``AveragePooling2D((1, pool_w), padding='same')`` over the time axis.

    TF's SAME average pooling divides by the number of *real* elements, which is
    ``count_include_pad=False`` plus ``ceil_mode=True`` in torch.
    """
    return F.avg_pool2d(x, kernel_size=(1, pool_w), stride=(1, pool_w),
                        ceil_mode=True, count_include_pad=False)
