#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""小规模 sanity training：用 2-4 个 source 被试、1-3 epoch 训练 minimal CAP-EEGNet。

目的（训练前准备，不是正式训练）：确认端到端可跑通——
  - 数据流 .npz -> SHUTrialDataset -> DataLoader -> CAP-EEGNet 无 shape 错误；
  - 训练 loss 随 epoch 下降；
  - 打印并记录 device / CUDA 可用性（GPU 可用与否）。

数据入口 = processed_manifest.csv 里 status==ok 的 .npz（绝不读 derivatives 的 .mat）。
**必须经 srun/sbatch 在计算节点跑，禁止登录节点**（见 50-server-slurm 规则）。

用法：
  python scripts/sanity_train.py --split splits/cap_eegnet_4110_seed2026.json \
      --n-subjects 3 --epochs 3 --out outputs/sanity_check/sanity_check_metrics.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data.shu_dataset import SHUTrialDataset  # noqa: E402
from src.data.splits import load_split  # noqa: E402
from src.models.cap_eegnet import CAPEEGNet, CAPEEGNetConfig  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.io import save_json  # noqa: E402
from src.utils.logging_utils import get_logger  # noqa: E402
from src.utils.paths import load_paths  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logger = get_logger("sanity_train")


def _cuda_report() -> dict:
    """汇总 torch / CUDA 信息（GPU 可用与否）。"""
    info = {
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,          # None == CPU-only wheel
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if info["cuda_available"]:
        info["device_name"] = torch.cuda.get_device_name(0)
    return info


def build_config(cfg: dict) -> CAPEEGNetConfig:
    """从 train_cross_subject.yaml 的 model 段取 encoder 超参，组成 minimal config。"""
    data = cfg.get("data", {})
    enc = (cfg.get("model", {}) or {}).get("encoder", {})
    return CAPEEGNetConfig(
        n_channels=data.get("n_channels", 58),
        n_times=data.get("n_times", 1000),
        n_classes=data.get("n_classes", 2),
        F1=enc.get("F1", 8),
        D=enc.get("D", 2),
        F2=enc.get("F2", 16),
        kernel_length=enc.get("kernel_length", 64),
        dropout=enc.get("dropout", 0.25),
        # minimal：三个高级 head 关闭。
        use_adapter=False, use_prototype=False, use_confidence=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal CAP-EEGNet sanity training.")
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--config", default="configs/train_cross_subject.yaml")
    ap.add_argument("--split", default="splits/cap_eegnet_4110_seed2026.json")
    ap.add_argument("--n-subjects", type=int, default=3, help="用前 N 个 source 被试（建议 2-4）。")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/sanity_check/sanity_check_metrics.json")
    args = ap.parse_args()

    if not (1 <= args.n_subjects <= 6):
        raise ValueError("sanity 只用少量被试（建议 2-4），n_subjects 过大请改用正式训练入口。")

    set_seed(args.seed)
    t_start = time.time()

    P = load_paths(PROJECT_ROOT / args.paths, require_raw=False)
    cfg = load_config(PROJECT_ROOT / args.config)
    split = load_split(args.split if Path(args.split).is_absolute() else PROJECT_ROOT / args.split)

    cuda = _cuda_report()
    device = torch.device("cuda" if cuda["cuda_available"] else "cpu")
    logger.info("CUDA: %s", cuda)
    if not cuda["cuda_available"]:
        logger.warning("CUDA 不可用 -> 本次 sanity 在 CPU 上跑（torch_cuda_build=%s）。"
                       " 正式 GPU 训练前需装 cu118 版 torch，见 docs/ENVIRONMENT.md。",
                       cuda["torch_cuda_build"])

    # —— 数据：取前 N 个 source 被试的 ok session ——
    subjects: List[str] = list(split["source_subjects"])[: args.n_subjects]
    n_files = len(subjects) * 3
    ds = SHUTrialDataset.from_manifest(
        P.processed_manifest, subjects=subjects, statuses=("ok",),
        cache_size=max(4, n_files),
    )
    logger.info("sanity 数据：subjects=%s | %d npz | %d trials",
                subjects, len(ds.npz_paths), len(ds))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        drop_last=False)

    # —— 模型 / 优化器 ——
    model_cfg = build_config(cfg)
    model = CAPEEGNet(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    logger.info("模型 CAP-EEGNet(minimal) | feature_dim=%d | params=%d | device=%s",
                model.feature_dim, n_params, device)

    # —— 训练循环 ——
    epoch_metrics = []
    first_batch_shapes = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_correct = total_n = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if first_batch_shapes is None:
                # 记录统一格式：DataLoader 给 [B,C,T]，模型内部 4D 化为 [B,1,C,T]。
                first_batch_shapes = {
                    "dataloader_x": list(xb.shape),
                    "model_internal_x": [xb.shape[0], 1, xb.shape[1], xb.shape[2]],
                    "y_dtype": str(yb.dtype),
                }
            optimizer.zero_grad()
            logits = model(xb)["logits"]
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = yb.size(0)
            total_loss += float(loss.item()) * bs
            total_correct += int((logits.argmax(1) == yb).sum().item())
            total_n += bs

        m = {"epoch": epoch,
             "train_loss": total_loss / total_n,
             "train_acc": total_correct / total_n}
        epoch_metrics.append(m)
        logger.info("epoch %d/%d | loss=%.4f | acc=%.4f",
                    epoch, args.epochs, m["train_loss"], m["train_acc"])

    # —— 判定 ——
    losses = [m["train_loss"] for m in epoch_metrics]
    loss_decreased = bool(len(losses) >= 2 and losses[-1] < losses[0])
    result = {
        "status": "ok",
        "purpose": "minimal CAP-EEGNet sanity training (not formal training)",
        "split_file": str(args.split),
        "subjects": subjects,
        "n_npz": len(ds.npz_paths),
        "n_trials": len(ds),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "model": {"name": "cap_eegnet_minimal", "feature_dim": model.feature_dim,
                  "n_params": n_params},
        "device": str(device),
        "cuda": cuda,
        "batch_shapes": first_batch_shapes,
        "epoch_metrics": epoch_metrics,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "loss_decreased": loss_decreased,
        "elapsed_sec": round(time.time() - t_start, 1),
    }

    out_path = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    save_json(result, out_path)
    logger.info("sanity 结果写入 %s", out_path)
    logger.info("loss: %.4f -> %.4f | decreased=%s | elapsed=%.1fs",
                losses[0], losses[-1], loss_decreased, result["elapsed_sec"])

    if not loss_decreased:
        logger.warning("训练 loss 未下降（sanity 期望下降）；请检查 lr/epoch/数据。")


if __name__ == "__main__":
    main()
