#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SHUTrialDataset smoke test：从 processed_manifest.csv + split JSON 构造 source/target
dataset，校验张量契约，并喂一个 batch 进 minimal CAP-EEGNet。

校验项：
  - 数据入口只读 status==ok 的 per-session .npz（绝不读 derivatives 的 .mat）。
  - 单样本 x = [58, 1000] float32；DataLoader batch x = [B, 58, 1000]（统一格式：
    模型内部再 unsqueeze 成 [B,1,58,1000]）。
  - y dtype == torch.long；label 仅 {0,1}。
  - minimal CAP-EEGNet forward(batch) -> logits [B, 2]，且内部 4D 化为 [B,1,58,1000]。

需要加载 .npz，**经 srun 在计算节点跑**（见 50-server-slurm 规则），不要在登录节点跑。
运行：python tests/test_dataset_smoke.py [--split splits/cap_eegnet_4110_seed2026.json]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data.shu_dataset import SHUTrialDataset  # noqa: E402
from src.data.splits import load_split  # noqa: E402
from src.models.cap_eegnet import CAPEEGNet, CAPEEGNetConfig  # noqa: E402
from src.utils.paths import load_paths  # noqa: E402


def _build(manifest: Path, subjects: List[str]) -> SHUTrialDataset:
    # cache_size 设为文件数，避免 LRU 抖动反复解压 .npz。
    n_files = len(subjects) * 3
    return SHUTrialDataset.from_manifest(
        manifest, subjects=subjects, statuses=("ok",), cache_size=max(4, n_files)
    )


def run(split_path: str, paths_cfg: str = "configs/paths.yaml",
        n_subjects: int = 2, batch_size: int = 64) -> None:
    P = load_paths(PROJECT_ROOT / paths_cfg, require_raw=False)
    split = load_split(split_path if Path(split_path).is_absolute()
                       else PROJECT_ROOT / split_path)

    src_subs = list(split["source_subjects"])[:n_subjects]
    tgt_subs = list(split["target_subjects"])[:n_subjects]
    print(f"== SHUTrialDataset smoke test ==  split={Path(split_path).name}")
    print(f"manifest = {P.processed_manifest}")
    print(f"source subset = {src_subs}\ntarget subset = {tgt_subs}\n")

    failures: List[str] = []
    for role, subs in (("source", src_subs), ("target", tgt_subs)):
        ds = _build(P.processed_manifest, subs)

        # 入口只能是 .npz —— 逐个 npz_path 校验，确认绝不碰 .mat。
        for p in ds.npz_paths:
            if p.suffix != ".npz":
                failures.append(f"{role}: 非 .npz 入口 {p}")
            if "derivatives" in str(p):
                failures.append(f"{role}: 入口疑似指向 derivatives：{p}")

        # 单样本契约
        x0, y0 = ds[0]
        if tuple(x0.shape) != (58, 1000):
            failures.append(f"{role}: 单样本 x.shape={tuple(x0.shape)} != (58,1000)")
        if x0.dtype != torch.float32:
            failures.append(f"{role}: 单样本 x.dtype={x0.dtype} != float32")

        # DataLoader batch 契约
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
        xb, yb = next(iter(loader))
        if tuple(xb.shape) != (batch_size, 58, 1000):
            failures.append(f"{role}: batch x.shape={tuple(xb.shape)} != ({batch_size},58,1000)")
        if yb.dtype != torch.long:
            failures.append(f"{role}: y.dtype={yb.dtype} != torch.long")
        uniq = sorted(set(int(v) for v in yb.tolist()))
        if not set(uniq).issubset({0, 1}):
            failures.append(f"{role}: label 超出 {{0,1}}：{uniq}")

        # label 全量分布（从 dataset.labels，便宜）
        labels = ds.labels
        n0 = sum(1 for v in labels if v == 0)
        n1 = sum(1 for v in labels if v == 1)

        # 喂进 minimal CAP-EEGNet
        model = CAPEEGNet(CAPEEGNetConfig())
        model.eval()
        with torch.no_grad():
            out = model(xb)
        logits = out["logits"]
        if tuple(logits.shape) != (batch_size, 2):
            failures.append(f"{role}: logits.shape={tuple(logits.shape)} != ({batch_size},2)")

        print(f"[{role}] {len(ds.npz_paths)} npz | {len(ds)} trials | label 0/1 = {n0}/{n1}")
        print(f"   item x: {tuple(x0.shape)} {x0.dtype} | batch x: {tuple(xb.shape)} "
              f"| y: {tuple(yb.shape)} {yb.dtype} | batch labels={uniq}")
        print(f"   model.feature_dim={model.feature_dim} | logits: {tuple(logits.shape)} "
              f"(internal 4D = [{batch_size},1,58,1000])")
        print()

    if failures:
        for f in failures:
            print("  !!", f)
        raise AssertionError("SHUTrialDataset smoke test 失败，见上方 !! 项。")
    print("SHUTrialDataset smoke test PASS ✓")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SHUTrialDataset smoke test.")
    ap.add_argument("--split", default="splits/cap_eegnet_4110_seed2026.json")
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--n-subjects", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    run(args.split, args.paths, args.n_subjects, args.batch_size)
