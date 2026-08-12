#!/usr/bin/env python3
"""Complete training template for models_eeg_foundation.

Usage:
  python train_template.py --data-path /path/to/data --model dualcd_s4_pos

Supports:
  - s4erp:         S4ERP (supervised baseline)
  - dualcd_s4_pos: UnifiedDINODualCD_S4_Pos (recommended for small datasets)
  - dualcd_s4_tp:  UnifiedDINODualCD_S4_Timepatch (interpretable)
  - dualcd_s4_flat: UnifiedDINODualCD_S4_Flatten (large datasets)
  - dualcd_tf:     UnifiedDINODualCD_Transformer (original baseline)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models_eeg_foundation import (
    S4ERP,
    UnifiedDINODualCD_S4_Flatten,
    UnifiedDINODualCD_S4_Pos,
    UnifiedDINODualCD_S4_Timepatch,
    UnifiedDINODualCD_Transformer,
)


# ── Config ───────────────────────────────────────────────────────────────────

class Config:
    """Dataset configuration. Modify for your dataset."""
    def __init__(self, num_channels, num_classes, seq_len, sampling_rate):
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.sampling_rate = sampling_rate


# ── Data Loading ─────────────────────────────────────────────────────────────

def normalize(x: np.ndarray) -> np.ndarray:
    """Per-sample z-score normalization. x: (N, T, C)"""
    m = x.mean(axis=1, keepdims=True)
    s = x.std(axis=1, keepdims=True).clip(min=1e-8)
    return (x - m) / s


def load_data(data_path: str):
    """Load data from .npy files.

    Expected files:
      data_path/X_train.npy  (N_train, T, C) or (N_train, C, T)
      data_path/y_train.npy  (N_train,)
      data_path/X_val.npy    (N_val, T, C) or (N_val, C, T)
      data_path/y_val.npy    (N_val,)
      data_path/X_test.npy   (N_test, T, C) or (N_test, C, T)
      data_path/y_test.npy   (N_test,)

    If files don't exist, tries train/val/test split from single arrays.
    """
    p = Path(data_path)

    if (p / "X_train.npy").exists():
        X_tr = np.load(p / "X_train.npy")
        y_tr = np.load(p / "y_train.npy")
        X_va = np.load(p / "X_val.npy")
        y_va = np.load(p / "y_val.npy")
        X_te = np.load(p / "X_test.npy")
        y_te = np.load(p / "y_test.npy")
    elif (p / "X.npy").exists():
        X = np.load(p / "X.npy")
        y = np.load(p / "y.npy")
        # Auto-split: 70/15/15
        n = len(y)
        idx = np.random.permutation(n)
        n_tr = int(0.7 * n)
        n_va = int(0.15 * n)
        X_tr, y_tr = X[idx[:n_tr]], y[idx[:n_tr]]
        X_va, y_va = X[idx[n_tr:n_tr+n_va]], y[idx[n_tr:n_tr+n_va]]
        X_te, y_te = X[idx[n_tr+n_va:]], y[idx[n_tr+n_va:]]
    else:
        raise FileNotFoundError(f"No data found at {data_path}")

    # Ensure (N, T, C) format
    if X_tr.ndim == 3 and X_tr.shape[1] < X_tr.shape[2]:
        X_tr = X_tr.transpose(0, 2, 1)
        X_va = X_va.transpose(0, 2, 1)
        X_te = X_te.transpose(0, 2, 1)

    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)


# ── Model Factory ────────────────────────────────────────────────────────────

def build_model(model_name: str, config, args):
    """Build model by name."""
    models = {
        "s4erp": lambda: S4ERP(config, d_model=128, n_layers=4, state_dim=8),
        "dualcd_s4_pos": lambda: UnifiedDINODualCD_S4_Pos(
            config, d_model=128, n_layers=4, state_dim=8,
            lambda_intra=args.lambda_intra, dino_out_dim=256,
            proto_k=5, teacher_momentum=0.996,
        ),
        "dualcd_s4_tp": lambda: UnifiedDINODualCD_S4_Timepatch(
            config, d_model=128, n_layers=4, state_dim=8,
            lambda_intra=args.lambda_intra, dino_out_dim=256,
            proto_k=5, teacher_momentum=0.996,
        ),
        "dualcd_s4_flat": lambda: UnifiedDINODualCD_S4_Flatten(
            config, d_model=128, n_layers=4, state_dim=8,
            lambda_intra=args.lambda_intra, dino_out_dim=256,
            proto_k=5, teacher_momentum=0.996,
        ),
        "dualcd_tf": lambda: UnifiedDINODualCD_Transformer(
            config, d_model=128, n_layers=6, n_heads=8,
            lambda_intra=args.lambda_intra, dino_out_dim=256,
            proto_k=5, teacher_momentum=0.996,
        ),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    return models[model_name]()


# ── Training ─────────────────────────────────────────────────────────────────

def train_epoch_supervised(model, loader, optimizer, device):
    """Training epoch for S4ERP (supervised)."""
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out["logits"], y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n


def train_epoch_dualcd(model, loader, optimizer, device, epoch):
    """Training epoch for DualCD models."""
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss, parts = model.compute_loss(x, y, epoch=epoch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
        optimizer.step()
        model.update_ema()
        model.update_prototypes(x, y)
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model on a DataLoader."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for x, y in loader:
        x = x.to(device)
        if hasattr(model, 'compute_loss'):
            logits = model(x)
        else:
            logits = model(x)["logits"]
        probs = torch.softmax(logits, 1)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(y.numpy())
        all_probs.append(probs.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    try:
        auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    except ValueError:
        auc = 0.0
    return {"accuracy": acc, "macro_f1": f1, "auroc": auc}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train EEG/ERP foundation models")
    ap.add_argument("--data-path", required=True, help="Path to data directory")
    ap.add_argument("--model", default="dualcd_s4_pos",
                    choices=["s4erp", "dualcd_s4_pos", "dualcd_s4_tp",
                             "dualcd_s4_flat", "dualcd_tf"],
                    help="Model variant to train")
    ap.add_argument("--output-dir", default="results/")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None, help="Auto-set if omitted")
    ap.add_argument("--lambda-intra", type=float, default=0.5)
    ap.add_argument("--num-channels", type=int, required=True)
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--sampling-rate", type=float, default=200.0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    # Setup
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Data
    print(f"Loading data from {args.data_path}...")
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_data(args.data_path)
    X_tr, X_va, X_te = normalize(X_tr), normalize(X_va), normalize(X_te)
    print(f"  Train: {len(y_tr)}, Val: {len(y_va)}, Test: {len(y_te)}")
    print(f"  Shape: {X_tr.shape}, Channels: {args.num_channels}, Classes: {args.num_classes}")

    config = Config(args.num_channels, args.num_classes, args.seq_len, args.sampling_rate)

    # Model
    model = build_model(args.model, config, args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {args.model}, Parameters: {n_params:,}")

    # Optimizer
    lr = args.lr or (1e-3 if args.model == "s4erp" else 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)),
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_va), torch.LongTensor(y_va)),
        batch_size=args.batch_size,
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_te), torch.LongTensor(y_te)),
        batch_size=args.batch_size,
    )

    # Training loop
    is_supervised = (args.model == "s4erp")
    train_fn = train_epoch_supervised if is_supervised else train_epoch_dualcd

    best_f1, best_epoch, wait = -1.0, 0, 0
    history = []

    print(f"\nTraining {args.model} for {args.epochs} epochs (lr={lr})...")
    for epoch in range(args.epochs):
        t0 = time.time()
        if is_supervised:
            tr_loss = train_fn(model, train_loader, optimizer, device)
        else:
            tr_loss = train_fn(model, train_loader, optimizer, device, epoch)
        scheduler.step()

        val_m = evaluate(model, val_loader, device)
        te_m = evaluate(model, test_loader, device)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d} | loss={tr_loss:.4f} | "
              f"val_f1={val_m['macro_f1']:.4f} te_f1={te_m['macro_f1']:.4f} | "
              f"{elapsed:.0f}s")

        history.append({
            "epoch": epoch, "train_loss": tr_loss,
            "val": val_m, "test": te_m, "time": elapsed,
        })

        # Early stopping
        if val_m["macro_f1"] > best_f1:
            best_f1 = val_m["macro_f1"]
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), out / "best_model.pt")
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {epoch} (best={best_epoch})")
                break

    # Final evaluation
    model.load_state_dict(torch.load(out / "best_model.pt", map_location=device))
    test_m = evaluate(model, test_loader, device)

    # Save results
    summary = {
        "model": args.model,
        "dataset": args.data_path,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
        "parameters": n_params,
        "test_accuracy": test_m["accuracy"],
        "test_macro_f1": test_m["macro_f1"],
        "test_auroc": test_m["auroc"],
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone! Best epoch: {best_epoch}")
    print(f"  Test Acc: {test_m['accuracy']:.4f}")
    print(f"  Test F1:  {test_m['macro_f1']:.4f}")
    print(f"  Test AUC: {test_m['auroc']:.4f}")
    print(f"  Results saved to: {out}")


if __name__ == "__main__":
    main()
