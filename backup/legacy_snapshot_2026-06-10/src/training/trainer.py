"""Generic single-model trainer shared by every model + protocol (fair comparison).

One trainer trains EEGNet / DeepConvNet / FBCNet / CAP-EEGNet identically:
  * loss = CrossEntropy(logits, y)  [+ confidence_weight * BCE(confidence, correct)]
    The confidence term is added ONLY when the model emits a learned confidence
    (CAP-EEGNet v1); ``correct`` = (argmax(logits) == y), detached. Baselines train
    with plain CE, so nothing about the comparison is unfair.
  * early stopping on a held-out validation slice (monitored: val loss). The best
    (lowest val-loss) state is restored before returning.

Tensor convention: loaders yield ``X = [B, C, T]`` (C=58, T=1000) float32 and
``y = [B]`` int64 in {0,1}. Models add the singleton ``[B,1,C,T]`` internally and
return ``{"logits", "features", "confidence"}``.

No data paths or protocol logic live here — see src/evaluation/session_protocols.py.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def make_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float):
    """Build an optimizer by name (adam | adamw | sgd)."""
    name = (name or "adam").lower()
    params = model.parameters()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer '{name}' (use adam | adamw | sgd).")


def _confidence_loss(confidence: torch.Tensor, logits: torch.Tensor,
                     y: torch.Tensor, bce: nn.BCELoss) -> torch.Tensor:
    """BCE between learned confidence and prediction correctness (calibration-style)."""
    with torch.no_grad():
        correct = (logits.argmax(dim=1) == y).float()
    conf = torch.clamp(confidence, 1e-6, 1.0 - 1e-6)
    return bce(conf, correct)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    *,
    max_epochs: int,
    lr: float,
    weight_decay: float = 0.0,
    optimizer: str = "adam",
    early_stopping_patience: int = 20,
    device: torch.device | str = "cpu",
    confidence_weight: float = 0.0,
    logger=None,
) -> Dict[str, object]:
    """Train one model with early stopping; restore + return the best state.

    Returns dict: best_val_loss, best_epoch, epochs_run, history (per-epoch
    train/val loss). If ``val_loader`` is None, trains the full ``max_epochs`` and
    keeps the final state (no early stopping).
    """
    device = torch.device(device)
    model.to(device)
    opt = make_optimizer(model, optimizer, lr, weight_decay)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCELoss()

    best_val = math.inf
    best_state = None
    best_epoch = -1
    patience = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        tot_loss = tot_n = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            out = model(xb)
            logits = out["logits"]
            loss = ce(logits, yb)
            conf = out.get("confidence")
            if conf is not None and confidence_weight > 0:
                loss = loss + confidence_weight * _confidence_loss(conf, logits, yb, bce)
            loss.backward()
            opt.step()
            bs = yb.size(0)
            tot_loss += float(loss.item()) * bs
            tot_n += bs
        train_loss = tot_loss / max(tot_n, 1)

        val_loss = float("nan")
        if val_loader is not None:
            val_loss = _val_loss(model, val_loader, device, ce)
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
                best_epoch = epoch
                patience = 0
            else:
                patience += 1
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if logger is not None:
            logger.debug("epoch %d/%d train_loss=%.4f val_loss=%.4f patience=%d",
                         epoch, max_epochs, train_loss, val_loss, patience)
        if val_loader is not None and patience >= early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = history[-1]["epoch"] if history else 0
        best_val = history[-1]["val_loss"] if history else float("nan")

    return {
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "epochs_run": history[-1]["epoch"] if history else 0,
        "history": history,
    }


@torch.no_grad()
def _val_loss(model: nn.Module, loader: DataLoader, device: torch.device, ce: nn.Module) -> float:
    model.eval()
    tot = n = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)["logits"]
        tot += float(ce(logits, yb).item()) * yb.size(0)
        n += yb.size(0)
    return tot / max(n, 1)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device | str = "cpu",
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Run inference; return (y_true, y_pred, probs[N,C], confidence[N] or None).

    ``probs`` are softmax probabilities (used for AUC/NLL/Brier/ECE regardless of
    whether the model has a learned confidence head).
    """
    device = torch.device(device)
    model.to(device)
    model.eval()
    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    cs: List[np.ndarray] = []
    has_conf = True
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        out = model(xb)
        prob = torch.softmax(out["logits"], dim=1).cpu().numpy()
        ps.append(prob)
        ys.append(yb.numpy())
        conf = out.get("confidence")
        if conf is None:
            has_conf = False
        else:
            cs.append(conf.detach().cpu().numpy())
    y_true = np.concatenate(ys)
    probs = np.concatenate(ps)
    y_pred = probs.argmax(axis=1)
    confidence = np.concatenate(cs) if (has_conf and cs) else None
    return y_true, y_pred, probs, confidence
