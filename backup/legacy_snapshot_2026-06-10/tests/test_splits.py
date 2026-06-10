#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Split smoke test：校验 41/10 subject-wise split JSON 的正确性并打印统计。

校验项（铁律）：
  - source = 41，target = 10。
  - source / target 无交集（无被试泄漏）。
  - target 每个被试都有 ses-01/ses-02/ses-03 且 status == ok。
  - source / target 实际使用的 session 全部 status == ok。
  - excluded_sessions 全部确实是 manifest 中 status != ok 的 session。

并打印：每个 split 的 trial 数、session 数、label 分布（0/1）。
trial 数与 label 分布来自 processed_manifest.csv 的计数列（不加载 .npz，登录节点安全）。

运行：
  python tests/test_splits.py                       # 校验 splits/ 下所有 cap_eegnet_4110_*.json
  python tests/test_splits.py --run-id cap_eegnet_4110
可被 pytest 收集（test_all_splits_valid）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import (  # noqa: E402
    REQUIRED_SESSIONS,
    SubjectSessionIndex,
    load_split,
    read_processed_manifest,
)
from src.utils.paths import load_paths  # noqa: E402

EXPECTED_SOURCE = 41
EXPECTED_TARGET = 10


def _label_distribution(idx: SubjectSessionIndex, subj_sessions: Dict[str, List[str]]):
    """对给定 {subject:[sessions]} 汇总 trial 数与 label 0/1 计数（来自 manifest 计数列）。"""
    n_sessions = 0
    n_trials = l0 = l1 = 0
    for subj, sessions in subj_sessions.items():
        for ses in sessions:
            c = idx.session_label_counts(subj, ses)
            n_sessions += 1
            n_trials += c.get("n_trials", 0)
            l0 += c.get("label_0_count", 0)
            l1 += c.get("label_1_count", 0)
    return {"n_sessions": n_sessions, "n_trials": n_trials, "label_0": l0, "label_1": l1}


def validate_split(split: dict, idx: SubjectSessionIndex) -> List[str]:
    """返回失败信息列表（空 = 通过）。"""
    errors: List[str] = []
    source = list(split["source_subjects"])
    target = list(split["target_subjects"])

    if len(source) != EXPECTED_SOURCE:
        errors.append(f"source={len(source)} != {EXPECTED_SOURCE}")
    if len(target) != EXPECTED_TARGET:
        errors.append(f"target={len(target)} != {EXPECTED_TARGET}")

    overlap = set(source) & set(target)
    if overlap:
        errors.append(f"source/target 交集非空: {sorted(overlap)}")

    # 唯一性
    if len(set(source)) != len(source):
        errors.append("source 内有重复被试")
    if len(set(target)) != len(target):
        errors.append("target 内有重复被试")

    # target 每个被试 3 session 全 ok
    for s in target:
        for ses in REQUIRED_SESSIONS:
            if idx.status.get(s, {}).get(ses) != "ok":
                errors.append(f"target {s}/{ses} 不是 ok（target 必须 3 session 全 ok）")

    # source/target 实际使用 session 全部 ok
    for role, mapping in (("source", split["source_train_sessions"]),
                          ("target", split["target_sessions"])):
        for s, sessions in mapping.items():
            for ses in sessions:
                if idx.status.get(s, {}).get(ses) != "ok":
                    errors.append(f"{role} {s}/{ses} 被使用但 status != ok")

    # excluded_sessions 都是真正的非 ok session
    for tag in split["excluded_sessions"]:
        s, ses = tag.split("/")
        if idx.status.get(s, {}).get(ses) == "ok":
            errors.append(f"excluded {tag} 实际是 ok，不应被排除")

    # source ok session 数应与 manifest 一致（被试维度）
    for s in source:
        expect = idx.ok_sessions(s)
        got = list(split["source_train_sessions"].get(s, []))
        if sorted(got) != sorted(expect):
            errors.append(f"source {s} 使用 session {got} != manifest ok {expect}")

    return errors


def run(run_id: str = "cap_eegnet_4110", paths_cfg: str = "configs/paths.yaml") -> None:
    P = load_paths(PROJECT_ROOT / paths_cfg, require_raw=False)
    idx = SubjectSessionIndex(read_processed_manifest(P.processed_manifest))

    split_files = sorted(Path(P.splits_dir).glob(f"{run_id}_seed*.json"))
    if not split_files:
        raise FileNotFoundError(
            f"在 {P.splits_dir} 找不到 {run_id}_seed*.json。先运行 scripts/make_splits.py。"
        )

    print(f"== Split smoke test ==  manifest={P.processed_manifest}")
    print(f"找到 {len(split_files)} 个 split 文件\n")

    all_ok = True
    for f in split_files:
        split = load_split(f)
        errors = validate_split(split, idx)
        src_dist = _label_distribution(idx, split["source_train_sessions"])
        tgt_dist = _label_distribution(idx, split["target_sessions"])

        status = "PASS" if not errors else "FAIL"
        all_ok = all_ok and not errors
        print(f"[{status}] {f.name}  (seed={split['seed']})")
        print(f"   source: {len(split['source_subjects'])} subjects | "
              f"{src_dist['n_sessions']} sessions | {src_dist['n_trials']} trials | "
              f"label 0/1 = {src_dist['label_0']}/{src_dist['label_1']}")
        print(f"   target: {len(split['target_subjects'])} subjects | "
              f"{tgt_dist['n_sessions']} sessions | {tgt_dist['n_trials']} trials | "
              f"label 0/1 = {tgt_dist['label_0']}/{tgt_dist['label_1']}")
        print(f"   excluded(failed) sessions: {split['excluded_sessions']}")
        print(f"   target subjects: {split['target_subjects']}")
        if errors:
            for e in errors:
                print(f"   !! {e}")
        print()

    if not all_ok:
        raise AssertionError("有 split 未通过校验，见上方 FAIL 项。")
    print("ALL SPLITS PASS ✓")


def test_all_splits_valid() -> None:
    """pytest 入口：所有 split 必须通过校验（无异常 = 通过）。"""
    run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Split smoke test.")
    ap.add_argument("--run-id", default="cap_eegnet_4110")
    ap.add_argument("--paths", default="configs/paths.yaml")
    args = ap.parse_args()
    run(run_id=args.run_id, paths_cfg=args.paths)
