"""按被试(subject-wise)划分工具：41 source / 10 target。

铁律（见 .cursor/rules/30-model-experiments 与 docs/EXPERIMENT_PROTOCOL.md）：
  - 必须按被试划分，绝不按 session / trial 划分。
  - target 被试绝不进入 source 训练（无泄漏）。
  - 每个划分持久化到 configs/paths.yaml 指定的 splits 目录（JSON），可复现。

本项目特有约束（来自 processed_manifest.csv 的 status 列）：
  - 训练/评估的数据入口 = status == "ok" 的 per-session .npz；derivatives 的 .mat 只做
    QC 对照，绝不作为入口（本模块只读 processed_manifest.csv，不碰 .mat）。
  - target 被试必须 **3 个 session 都 ok**，因为后续要做 Session 1 微调 +
    Session 2/3 在线 test-then-update，缺任何一个 session 都无法完整跑该流程。
  - source 被试 = 其余 41 个被试，**允许包含有 failed session 的被试**，但训练时只用其
    status == "ok" 的 session（有 failed session 的被试因此不能当 target，会被强制留在 source）。
  - failed session 一律记入 `excluded_sessions`，不进入任何训练/评估。

数据事实（2026-06-05 全量 eog_ecg_clean 预处理后）：51 被试 / 153 session，148 ok / 5 failed；
5 个 failed 属于 4 个被试（sub-023/024/026/032），均为原始 trigger/试次<200，与降噪无关。
"""

from __future__ import annotations

import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..utils.paths import ses_id, sub_id

# 每个被试期望具备的 session（2C 数据集：每被试 3 个 session）。
REQUIRED_SESSIONS: Tuple[str, ...] = ("ses-01", "ses-02", "ses-03")
OK_STATUS = "ok"

# 当前阶段固定协议：target 用 Session 1 微调，Session 2/3 在线 test-then-update。
DEFAULT_TARGET_FINETUNE_SESSIONS: Tuple[int, ...] = (1,)
DEFAULT_TARGET_ONLINE_SESSIONS: Tuple[int, ...] = (2, 3)


# --------------------------------------------------------------------------- #
# manifest 读取与按被试汇总
# --------------------------------------------------------------------------- #
def read_processed_manifest(manifest_path: str | Path) -> List[Dict[str, str]]:
    """读取 processed_manifest.csv（每行一个 session），返回行 dict 列表。"""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"processed manifest 不存在: {manifest_path}。先运行 scripts/preprocess_all.py 生成。"
        )
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"processed manifest 为空: {manifest_path}")
    return rows


class SubjectSessionIndex:
    """从 processed_manifest.csv 汇总出的「被试 → session → status」索引。

    提供后续 split 需要的全部判断：哪些被试 3 session 全 ok（可当 target）、
    每个被试的 ok session、全局 failed/excluded session 等。
    """

    def __init__(self, rows: Sequence[Dict[str, str]],
                 required_sessions: Sequence[str] = REQUIRED_SESSIONS):
        self.required_sessions = tuple(required_sessions)
        # subject -> {session -> status}
        self.status: Dict[str, Dict[str, str]] = {}
        # subject -> {session -> (label_0_count, label_1_count, n_trials)} 仅 ok 行有意义
        self._counts: Dict[str, Dict[str, Dict[str, int]]] = {}
        for r in rows:
            subj = sub_id(r["subject_id"])
            sess = ses_id(r["session_id"])
            self.status.setdefault(subj, {})[sess] = (r.get("status") or "").strip()
            self._counts.setdefault(subj, {})[sess] = {
                "n_trials": _to_int(r.get("n_trials")),
                "label_0_count": _to_int(r.get("label_0_count")),
                "label_1_count": _to_int(r.get("label_1_count")),
            }

    @property
    def all_subjects(self) -> List[str]:
        return sorted(self.status)

    def ok_sessions(self, subject: str) -> List[str]:
        """该被试 status == ok 的 session（排序）。"""
        s = self.status.get(sub_id(subject), {})
        return sorted(k for k, v in s.items() if v == OK_STATUS)

    def failed_sessions(self, subject: str) -> List[str]:
        """该被试 status != ok 的 session（排序）。"""
        s = self.status.get(sub_id(subject), {})
        return sorted(k for k, v in s.items() if v != OK_STATUS)

    def is_fully_ok(self, subject: str) -> bool:
        """该被试是否「所有 required session 都存在且 ok」——可当 target 的条件。"""
        s = self.status.get(sub_id(subject), {})
        return all(s.get(ses) == OK_STATUS for ses in self.required_sessions)

    def eligible_target_subjects(self) -> List[str]:
        """可作为 target 的被试：required session 全部 ok。"""
        return [s for s in self.all_subjects if self.is_fully_ok(s)]

    def excluded_sessions(self) -> List[Dict[str, str]]:
        """全局被排除（非 ok）的 session 明细（subject/session/status）。"""
        out: List[Dict[str, str]] = []
        for subj in self.all_subjects:
            for sess in sorted(self.status[subj]):
                st = self.status[subj][sess]
                if st != OK_STATUS:
                    out.append({"subject": subj, "session": sess, "status": st})
        return out

    def session_label_counts(self, subject: str, session: str) -> Dict[str, int]:
        return self._counts.get(sub_id(subject), {}).get(ses_id(session), {})


def _to_int(v) -> int:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------- #
# 核心：subject-wise 划分
# --------------------------------------------------------------------------- #
def make_subject_wise_split(
    manifest_path: str | Path,
    *,
    seed: int,
    n_source: int = 41,
    n_target: int = 10,
    required_sessions: Sequence[str] = REQUIRED_SESSIONS,
    variant: str = "eog_ecg_clean",
    run_id: str = "cap_eegnet_4110",
    target_finetune_sessions: Sequence[int] = DEFAULT_TARGET_FINETUNE_SESSIONS,
    target_online_sessions: Sequence[int] = DEFAULT_TARGET_ONLINE_SESSIONS,
) -> Dict[str, object]:
    """生成一个 41 source / 10 target 的 subject-wise 划分（按被试，不按 trial）。

    规则：
      1. target = 从「3 session 全 ok」的被试里随机抽 n_target 个（random.Random(seed)）。
      2. source = 其余全部被试（含有 failed session 的被试），共 n_source 个。
      3. 训练/评估只用 status == ok 的 session；failed session 记入 excluded_sessions。

    返回一个可直接 json.dump 的 dict（结构见模块文档 / save_split）。
    """
    rows = read_processed_manifest(manifest_path)
    idx = SubjectSessionIndex(rows, required_sessions=required_sessions)

    all_subjects = idx.all_subjects
    n_total = len(all_subjects)
    if n_source + n_target != n_total:
        raise ValueError(
            f"n_source({n_source}) + n_target({n_target}) = {n_source + n_target} "
            f"必须等于 manifest 中的被试总数 {n_total}。"
        )

    eligible = idx.eligible_target_subjects()
    if len(eligible) < n_target:
        raise ValueError(
            f"可作 target 的被试（3 session 全 ok）只有 {len(eligible)} 个，不足 n_target={n_target}。"
            f" 不合格被试={[s for s in all_subjects if s not in set(eligible)]}"
        )

    # 仅用本地 RNG，避免污染全局随机状态；排序后再抽样保证跨平台可复现。
    rng = random.Random(seed)
    target_subjects = sorted(rng.sample(sorted(eligible), n_target))
    target_set = set(target_subjects)
    source_subjects = [s for s in all_subjects if s not in target_set]

    if len(source_subjects) != n_source:
        raise ValueError(
            f"source 被试数 {len(source_subjects)} != 期望 {n_source}（内部不一致）。"
        )
    # 双保险：无交集。
    overlap = target_set & set(source_subjects)
    if overlap:
        raise ValueError(f"source/target 交集非空（泄漏）: {sorted(overlap)}")

    # excluded（failed）session：全部 failed session 必落在 source（target 已保证全 ok）。
    excluded = idx.excluded_sessions()

    # 记录每个被试实际使用的 ok session（训练入口）。
    source_ok_sessions = {s: idx.ok_sessions(s) for s in source_subjects}
    target_sessions = {s: idx.ok_sessions(s) for s in target_subjects}

    n_source_ok = sum(len(v) for v in source_ok_sessions.values())
    n_target_ok = sum(len(v) for v in target_sessions.values())

    split: Dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "seed": int(seed),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest_path": str(Path(manifest_path)),
        "variant": variant,
        "split_policy": (
            "subject-wise 41 source / 10 target; target subjects require all 3 sessions "
            "status=ok; source may include subjects with failed sessions but training uses "
            "ok sessions only; failed sessions go to excluded_sessions; NEVER trial/session-wise"
        ),
        "n_total_subjects": n_total,
        "n_source_subjects": len(source_subjects),
        "n_target_subjects": len(target_subjects),
        "required_sessions": list(required_sessions),
        # —— 划分主体 ——
        "source_subjects": source_subjects,
        "target_subjects": target_subjects,
        # failed/排除 session（不进入任何训练/评估）
        "excluded_sessions": [f"{e['subject']}/{e['session']}" for e in excluded],
        "excluded_sessions_detail": excluded,
        # target 协议
        "target_finetune_sessions": list(target_finetune_sessions),
        "target_online_sessions": list(target_online_sessions),
        # 实际使用的 ok session（便于 Dataset 构造与核对）
        "source_train_sessions": source_ok_sessions,
        "target_sessions": target_sessions,
        "counts": {
            "source_ok_sessions": n_source_ok,
            "target_ok_sessions": n_target_ok,
            "excluded_sessions": len(excluded),
            "source_ok_trials_est": n_source_ok * 200,
            "target_ok_trials_est": n_target_ok * 200,
        },
    }
    return split


# --------------------------------------------------------------------------- #
# 持久化
# --------------------------------------------------------------------------- #
def split_filename(run_id: str, seed: int) -> str:
    """统一的 split 文件名：<run_id>_seed<seed>.json。"""
    return f"{run_id}_seed{seed}.json"


def save_split(split: Dict[str, object], path: str | Path) -> Path:
    """把划分写入 JSON（供复现）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)
    return path


def load_split(path: str | Path) -> Dict[str, object]:
    """读取已保存的划分 JSON。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"split 文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_and_save_splits(
    manifest_path: str | Path,
    splits_dir: str | Path,
    seeds: Sequence[int],
    *,
    n_source: int = 41,
    n_target: int = 10,
    run_id: str = "cap_eegnet_4110",
    variant: str = "eog_ecg_clean",
    required_sessions: Sequence[str] = REQUIRED_SESSIONS,
) -> List[Path]:
    """为多个 seed 生成并保存 split JSON，返回写出的文件路径列表。"""
    splits_dir = Path(splits_dir)
    out_paths: List[Path] = []
    for seed in seeds:
        split = make_subject_wise_split(
            manifest_path,
            seed=seed,
            n_source=n_source,
            n_target=n_target,
            run_id=run_id,
            variant=variant,
            required_sessions=required_sessions,
        )
        out = save_split(split, splits_dir / split_filename(run_id, seed))
        out_paths.append(out)
    return out_paths
