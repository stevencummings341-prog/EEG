"""End-to-end trainer for the foundation-model track: resumable, 2 checkpoints only.

Why a second trainer next to ``trainer.py``
-------------------------------------------
``trainer.py`` drives the finished Phase 0-2c work and must keep producing bit-identical
results, so it is left untouched. The foundation-model track needs three things it does
not have:

1. **Non-CE losses.** The DualCD variants train with DINO + iBOT + DKoleo + CE +
   perturbation + prototype terms and need a teacher-EMA / prototype update *after* the
   optimizer step. Handled through the optional model hooks
   ``uses_custom_loss`` / ``training_step(x, y, epoch)`` / ``after_optimizer_step(x, y)``.
   A model without those hooks trains with plain cross-entropy, so EEGNet-family
   baselines can run through this trainer too (useful as an apples-to-apples control).
2. **Exactly two checkpoints per cell** — ``best.pt`` (best validation ``monitor``) and
   ``last.pt`` (most recent epoch). Nothing else is written, as required.
3. **Resume after preemption.** ``last.pt`` also carries optimizer + scheduler + RNG +
   history, so a Slurm job that dies mid-run continues from the next epoch instead of
   restarting. Checkpoints are written atomically (tmp file + ``os.replace``) so a kill
   during a write can never leave a truncated checkpoint behind.

Determinism under resume: when ``epoch_seed_base`` is given, every epoch reseeds with
``epoch_seed_base + epoch``. The data order of epoch *k* therefore does not depend on
whether the run was interrupted before it, so a resumed run stays comparable to an
uninterrupted one.

Tensor convention is unchanged: loaders yield ``X = [B, C, T]`` float32 and ``y = [B]``
int64; models return ``{"logits", "features", "confidence"}``.

Per-epoch train / val / test curves (2026-08-10)
------------------------------------------------
``train_eval_loader`` and ``test_loader`` are optional *monitoring* loaders, evaluated in
eval mode every epoch and written into ``history`` as ``train_eval`` / ``test``. They exist
so the three accuracy curves can be plotted side by side (advisor request).

**Model selection never sees them.** ``best.pt``, the early-stopping counter and
``best_score`` are computed from ``val_loader`` alone — see the guard right below the
per-epoch evaluation. Reporting a test curve that influenced training would be test-set
leakage, so the separation is enforced in code, not just by convention.

``train_eval_loader`` is a *capped subset* of the training set (the protocol layer decides
the cap) measured in eval mode, so it is directly comparable to the val/test curves instead
of being distorted by dropout. ``train_loss`` remains the true full-epoch training loss.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

BEST_CKPT = "best.pt"
LAST_CKPT = "last.pt"

EvaluateFn = Callable[[nn.Module, DataLoader, torch.device], Dict[str, float]]


@dataclass
class E2ESpec:
    """Training recipe for one end-to-end cell (shared by every model for fairness)."""

    batch_size: int = 64
    lr: Optional[float] = None          # None -> model.default_lr() if available, else 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"            # adam | adamw | sgd
    max_epochs: int = 100
    early_stopping_patience: int = 25   # <=0 disables early stopping
    monitor: str = "macro_f1"           # key of the validation metric dict
    monitor_mode: str = "max"           # max | min
    grad_clip_norm: float = 4.0         # <=0 disables clipping
    scheduler: str = "cosine"           # cosine | none
    num_workers: int = 2
    val_fraction: float = 0.2           # used by the protocol layer, kept here for the record
    normalization: str = "per_sample_zscore"   # applied by the protocol layer
    amp: bool = False                   # mixed precision on CUDA
    # Gradient accumulation: keeps the recipe's effective `batch_size` on models whose
    # activations do not fit in one step. None -> no accumulation (one step per batch).
    micro_batch_size: Optional[int] = None
    # Per-epoch monitoring curves (never used for model selection; see module docstring).
    curves: bool = False                # log per-epoch train/test metrics alongside val
    train_eval_max_trials: int = 2000   # cap of the train-accuracy monitoring subset

    @property
    def accum_steps(self) -> int:
        """How many micro-batches make up one optimizer step."""
        if not self.micro_batch_size or self.micro_batch_size >= self.batch_size:
            return 1
        return int(math.ceil(self.batch_size / self.micro_batch_size))

    def resolved_lr(self, model: nn.Module) -> float:
        if self.lr is not None:
            return float(self.lr)
        getter = getattr(model, "default_lr", None)
        return float(getter()) if callable(getter) else 1e-3


def make_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float):
    """Optimizer over the *trainable* parameters only.

    Filtering matters for the DualCD models: their teacher branch is frozen
    (``requires_grad=False``) and must never be touched by weight decay.
    """
    name = (name or "adamw").lower()
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("model has no trainable parameters")
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer '{name}' (use adam | adamw | sgd).")


# --------------------------------------------------------------------------- #
# Checkpoint IO (atomic)
# --------------------------------------------------------------------------- #
def _atomic_save(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _cpu_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def _rng_state() -> Dict[str, object]:
    import random
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
    }


def _restore_rng(state: Optional[Dict[str, object]]) -> None:
    if not state:
        return
    import random
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8))
        if state.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except (KeyError, RuntimeError, TypeError, ValueError):
        # A resumed run with a different device layout should continue, not crash;
        # only exact RNG replay is lost.
        pass


def load_best_state(ckpt_dir: str | Path, map_location: str | torch.device = "cpu"
                    ) -> Optional[Dict[str, torch.Tensor]]:
    """Model weights from ``best.pt`` (or None if absent)."""
    p = Path(ckpt_dir) / BEST_CKPT
    if not p.exists():
        return None
    return torch.load(p, map_location=map_location, weights_only=False)["model_state"]


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _default_eval(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Fallback validation metrics: loss + accuracy + balanced accuracy + macro F1."""
    from sklearn.metrics import balanced_accuracy_score, f1_score

    model.eval()
    losses, n = 0.0, 0
    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb_dev = yb.to(device, non_blocking=True)
        logits = model(xb)["logits"]
        losses += float(F.cross_entropy(logits, yb_dev).item()) * yb.size(0)
        n += yb.size(0)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        trues.append(yb.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    return {
        "val_loss": losses / max(n, 1),
        "accuracy": float((y_pred == y_true).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def _is_better(new: float, best: float, mode: str) -> bool:
    if not math.isfinite(new):
        return False
    return (new > best + 1e-9) if mode == "max" else (new < best - 1e-9)


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train_end_to_end(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    *,
    spec: E2ESpec,
    device: torch.device | str = "cpu",
    ckpt_dir: Optional[str | Path] = None,
    resume: bool = True,
    evaluate_fn: Optional[EvaluateFn] = None,
    epoch_seed_base: Optional[int] = None,
    run_signature: Optional[str] = None,
    train_eval_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    logger=None,
    log_every: int = 1,
) -> Dict[str, object]:
    """Train one cell to completion (or resume it) and return a run summary.

    Writes only ``<ckpt_dir>/best.pt`` and ``<ckpt_dir>/last.pt``. With
    ``val_loader=None`` there is no model selection: ``best`` tracks ``last``.

    ``train_eval_loader`` / ``test_loader`` are monitoring-only (per-epoch curves); they
    never affect ``best.pt`` or early stopping. See the module docstring.

    ``run_signature`` fingerprints everything that must not change between an
    interrupted run and its continuation (split membership, data dims, model params).
    Resuming onto a checkpoint with a different signature raises instead of silently
    continuing a model that was trained on other data.

    Returns keys: ``best_epoch``, ``best_score``, ``monitor``, ``epochs_run``,
    ``resumed_from_epoch``, ``early_stopped``, ``history`` (per-epoch dicts),
    ``final_val`` (last epoch's validation metrics), ``train_seconds``.
    """
    from ..utils.seed import set_seed

    device = torch.device(device)
    model.to(device)
    lr = spec.resolved_lr(model)
    opt = make_optimizer(model, spec.optimizer, lr, spec.weight_decay)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(int(spec.max_epochs), 1))
             if (spec.scheduler or "none").lower() == "cosine" else None)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(spec.amp) and device.type == "cuda")
    evaluate = evaluate_fn or _default_eval

    uses_custom = bool(getattr(model, "uses_custom_loss", False))
    after_step = getattr(model, "after_optimizer_step", None)
    mode = (spec.monitor_mode or "max").lower()

    ckpt_dir = Path(ckpt_dir) if ckpt_dir is not None else None
    start_epoch = 0
    best_score = -math.inf if mode == "max" else math.inf
    best_epoch = -1
    history: List[Dict[str, float]] = []
    resumed_from = None

    if resume and ckpt_dir is not None and (ckpt_dir / LAST_CKPT).exists():
        ck = torch.load(ckpt_dir / LAST_CKPT, map_location=device, weights_only=False)
        stored_sig = ck.get("run_signature")
        if run_signature is not None and stored_sig is not None and stored_sig != run_signature:
            raise RuntimeError(
                f"refusing to resume {ckpt_dir / LAST_CKPT}: it was trained under a different "
                f"configuration (stored signature {stored_sig}, current {run_signature}). "
                "The subject split, data dims or model params changed. Use a fresh --out / "
                "--ckpt-dir for the new setting, or pass --no-resume to retrain."
            )
        model.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["optimizer_state"])
        if sched is not None and ck.get("scheduler_state") is not None:
            sched.load_state_dict(ck["scheduler_state"])
        if ck.get("scaler_state") is not None:
            scaler.load_state_dict(ck["scaler_state"])
        start_epoch = int(ck["epoch"])
        best_score = float(ck.get("best_score", best_score))
        best_epoch = int(ck.get("best_epoch", -1))
        history = list(ck.get("history") or [])
        _restore_rng(ck.get("rng_state"))
        resumed_from = start_epoch
        if logger is not None:
            logger.info("resume: %s at epoch %d (best %s=%.4f @ epoch %d)",
                        ckpt_dir, start_epoch, spec.monitor, best_score, best_epoch)

    if start_epoch >= int(spec.max_epochs):
        if logger is not None:
            logger.info("nothing to do: already trained %d/%d epochs", start_epoch, spec.max_epochs)
        return {
            "best_epoch": best_epoch, "best_score": best_score, "monitor": spec.monitor,
            "epochs_run": start_epoch, "resumed_from_epoch": resumed_from,
            "early_stopped": False, "history": history,
            "final_val": (history[-1].get("val") if history else None),
            "train_seconds": 0.0, "lr": lr,
        }

    t_start = time.time()
    # Rebuild the early-stopping counter from the checkpoint instead of starting at 0,
    # otherwise every preemption silently grants another full patience window and a
    # resumed run trains longer than the same run would have uninterrupted.
    patience = max(0, start_epoch - best_epoch) if best_epoch > 0 else 0
    early_stopped = False
    epoch = start_epoch

    for epoch in range(start_epoch + 1, int(spec.max_epochs) + 1):
        if epoch_seed_base is not None:
            set_seed(int(epoch_seed_base) + epoch)
        model.train()
        t_epoch = time.time()
        tot_loss, tot_n = 0.0, 0
        parts_sum: Dict[str, float] = {}

        # With accumulation the loader yields micro-batches; `accum` of them make one step,
        # so the optimizer still sees the recipe's effective batch size. Note that
        # BatchNorm statistics are computed per micro-batch — the one semantic difference,
        # documented in the experiment config for the models that need it.
        accum = spec.accum_steps
        opt.zero_grad(set_to_none=True)
        pending = 0

        for step, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if uses_custom:
                loss, parts = model.training_step(xb, yb, epoch=epoch - 1)
            else:
                loss = F.cross_entropy(model(xb)["logits"], yb)
                parts = {"ce": float(loss.detach().item())}

            scaled = loss / accum if accum > 1 else loss
            if scaler.is_enabled():
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            pending += 1

            # Step on a full accumulation group, or on the epoch's ragged last group.
            if pending == accum or step == len(train_loader):
                if scaler.is_enabled():
                    if spec.grad_clip_norm and spec.grad_clip_norm > 0:
                        scaler.unscale_(opt)
                        nn.utils.clip_grad_norm_(model.parameters(), spec.grad_clip_norm)
                    scaler.step(opt)
                    scaler.update()
                else:
                    if spec.grad_clip_norm and spec.grad_clip_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), spec.grad_clip_norm)
                    opt.step()
                opt.zero_grad(set_to_none=True)
                pending = 0
                if callable(after_step):
                    after_step(xb, yb)

            bs = int(yb.size(0))
            tot_loss += float(loss.detach().item()) * bs
            tot_n += bs
            for k, v in parts.items():
                parts_sum[k] = parts_sum.get(k, 0.0) + float(v) * bs

        if sched is not None:
            sched.step()
        train_loss = tot_loss / max(tot_n, 1)
        loss_parts = {k: v / max(tot_n, 1) for k, v in parts_sum.items()}

        val_metrics = evaluate(model, val_loader, device) if val_loader is not None else {}
        # Monitoring-only curves. `score` below is derived from val_metrics ONLY: letting
        # train/test metrics reach model selection would be test-set leakage.
        train_metrics = (evaluate(model, train_eval_loader, device)
                         if train_eval_loader is not None else {})
        test_metrics = evaluate(model, test_loader, device) if test_loader is not None else {}

        score = float(val_metrics.get(spec.monitor, float("nan"))) if val_metrics else float("nan")
        row = {
            "epoch": epoch, "train_loss": train_loss, "lr": float(opt.param_groups[0]["lr"]),
            "loss_parts": loss_parts, "val": val_metrics,
            "epoch_seconds": round(time.time() - t_epoch, 2),
        }
        if train_metrics:
            row["train_eval"] = train_metrics
        if test_metrics:
            row["test"] = test_metrics
        history.append(row)

        improved = (val_loader is None) or _is_better(score, best_score, mode)
        if improved:
            best_score = score if val_loader is not None else float("nan")
            best_epoch = epoch
            patience = 0
            if ckpt_dir is not None:
                _atomic_save({
                    "model_state": _cpu_state(model), "epoch": epoch,
                    "monitor": spec.monitor, "score": best_score,
                    "val_metrics": val_metrics, "spec": asdict(spec),
                    "run_signature": run_signature,
                    "model_desc": (model.describe() if hasattr(model, "describe") else None),
                }, ckpt_dir / BEST_CKPT)
        else:
            patience += 1

        if ckpt_dir is not None:
            _atomic_save({
                "model_state": _cpu_state(model),
                "optimizer_state": opt.state_dict(),
                "scheduler_state": (sched.state_dict() if sched is not None else None),
                "scaler_state": (scaler.state_dict() if scaler.is_enabled() else None),
                "epoch": epoch, "best_score": best_score, "best_epoch": best_epoch,
                "monitor": spec.monitor, "history": history, "rng_state": _rng_state(),
                "spec": asdict(spec), "run_signature": run_signature,
                "model_desc": (model.describe() if hasattr(model, "describe") else None),
            }, ckpt_dir / LAST_CKPT)

        if logger is not None and (epoch % max(int(log_every), 1) == 0 or epoch == spec.max_epochs):
            extra = " ".join(f"{k}={v:.3f}" for k, v in sorted(loss_parts.items())) or "-"
            if train_metrics or test_metrics:
                # Compact 3-curve line: the numbers the advisor wants to see side by side.
                vtxt = "acc(train/val/test)=%s/%s/%s macro_f1(val)=%.4f" % (
                    f"{train_metrics.get('accuracy', float('nan')):.4f}",
                    f"{val_metrics.get('accuracy', float('nan')):.4f}",
                    f"{test_metrics.get('accuracy', float('nan')):.4f}",
                    val_metrics.get(spec.monitor, float("nan")),
                )
            else:
                vtxt = (" ".join(f"val_{k}={v:.4f}" for k, v in sorted(val_metrics.items()))
                        if val_metrics else "no-val")
            logger.info("epoch %3d/%d loss=%.4f [%s] %s patience=%d (%.1fs)",
                        epoch, spec.max_epochs, train_loss, extra, vtxt, patience,
                        row["epoch_seconds"])

        if (val_loader is not None and spec.early_stopping_patience
                and spec.early_stopping_patience > 0
                and patience >= int(spec.early_stopping_patience)):
            early_stopped = True
            if logger is not None:
                logger.info("early stop at epoch %d (best %s=%.4f @ epoch %d)",
                            epoch, spec.monitor, best_score, best_epoch)
            break

    return {
        "best_epoch": best_epoch, "best_score": best_score, "monitor": spec.monitor,
        "epochs_run": epoch, "resumed_from_epoch": resumed_from,
        "early_stopped": early_stopped, "history": history,
        "final_val": (history[-1]["val"] if history else None),
        "train_seconds": round(time.time() - t_start, 1), "lr": lr,
    }


@dataclass
class CellPaths:
    """Where one (model, fold, seed) cell keeps its two checkpoints and its result."""

    ckpt_dir: Path
    result_json: Path

    @property
    def best(self) -> Path:
        return self.ckpt_dir / BEST_CKPT

    @property
    def last(self) -> Path:
        return self.ckpt_dir / LAST_CKPT

    def is_complete(self) -> bool:
        """A cell counts as done only once its result JSON exists on disk."""
        return self.result_json.exists()
