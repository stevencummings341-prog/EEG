"""BatchNorm running-statistic adaptation (no-learning target adaptation).

``bn_statistics_adaptation`` keeps the SOURCE-trained weights frozen and only
refreshes the BatchNorm running mean/variance from the UNLABELED target X:

  * put ONLY the BN layers into ``train()`` mode (everything else stays
    ``eval()``), so a forward pass updates ``running_mean`` / ``running_var`` but
    dropout etc. stay deterministic;
  * forward the target X for ``n_passes`` (no labels, no loss, no ``backward``,
    NO optimizer — there is no optimizer object at all here);
  * restore ``eval()`` mode and predict on the target test set.

To get a clean estimate we (optionally) reset the BN running stats and set
``momentum=None`` so PyTorch accumulates a cumulative average over the forwarded
batches (one full pass => the exact target BN statistics).

This module never touches model parameters and never calls ``optimizer.step``;
the only state it changes is BN ``running_mean`` / ``running_var`` /
``num_batches_tracked``. Each method in the protocol builds its own fresh model,
so adapting BN here can never contaminate another method's model.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def count_bn_layers(model: nn.Module) -> int:
    """Number of BatchNorm layers in the model."""
    return sum(1 for m in model.modules() if isinstance(m, _BN_TYPES))


def _set_bn_train_only(model: nn.Module, reset: bool, cumulative: bool) -> List[float]:
    """Put BN layers in train mode (others eval); optionally reset stats. Returns saved momenta."""
    model.eval()
    saved_momentum: List[float] = []
    for m in model.modules():
        if isinstance(m, _BN_TYPES):
            m.train()
            saved_momentum.append(m.momentum)
            if reset:
                m.reset_running_stats()
            if cumulative:
                m.momentum = None   # cumulative moving average over forwarded batches
    return saved_momentum


def _restore_bn(model: nn.Module, saved_momentum: List[float]) -> None:
    i = 0
    for m in model.modules():
        if isinstance(m, _BN_TYPES):
            m.momentum = saved_momentum[i]
            i += 1
    model.eval()


@torch.no_grad()
def adapt_bn_statistics(
    model: nn.Module,
    target_loader: DataLoader,
    *,
    device: torch.device | str = "cpu",
    n_passes: int = 1,
    reset: bool = True,
    cumulative: bool = True,
) -> int:
    """Refresh BN running stats from the unlabeled target X (no grad/backward/optim).

    Returns the number of BN layers that were adapted. Raises if the model has no
    BN layers (so a silent no-op never masquerades as adaptation).
    """
    device = torch.device(device)
    model.to(device)
    n_bn = count_bn_layers(model)
    if n_bn == 0:
        raise ValueError("adapt_bn_statistics: model has no BatchNorm layers to adapt.")

    saved = _set_bn_train_only(model, reset=reset, cumulative=cumulative)
    try:
        for _ in range(max(1, int(n_passes))):
            for batch in target_loader:
                xb = batch[0] if isinstance(batch, (list, tuple)) else batch
                xb = xb.to(device, non_blocking=True)
                model(xb)   # forward only; updates BN running stats, no loss/backward
    finally:
        _restore_bn(model, saved)
    return n_bn
