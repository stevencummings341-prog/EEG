"""路径工具：封装数据集（只读）与项目内部目录的所有路径构造。

设计原则（见 .cursor/rules/）：
  - 数据集根目录只读，绝不写入。
  - 文件夹名 "2C dataset" 含空格，必须用 pathlib/os.path.join 拼接。
  - 被试数量从磁盘枚举，绝不写死（tsv 列 52，磁盘 51，README 说 53）。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

# 数据集根目录（只读）。可被函数参数覆盖。
DATASET_ROOT = Path("/share/workspace2/moto_imagination/WBCIC_SHU")

# 2C 数据集子目录（注意空格是数据集自带的）。
RAW_SUBDIR = "sourcedata/2C dataset"
DERIV_SUBDIR = "derivatives/2C dataset_processeddata"

# 项目根目录：本文件位于 <root>/src/utils/paths.py，故上溯三级。
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def dataset_root(root: Path | str | None = None) -> Path:
    """返回数据集根目录（默认常量，可覆盖）。"""
    return Path(root) if root is not None else DATASET_ROOT


def sub_id(subject: int | str) -> str:
    """规范化被试 id 为 'sub-XXX'（3 位零填充）。接受 1 / '1' / 'sub-001'。"""
    if isinstance(subject, str) and subject.startswith("sub-"):
        return subject
    return f"sub-{int(subject):03d}"


def ses_id(session: int | str) -> str:
    """规范化 session id 为 'ses-YY'（2 位零填充）。接受 1 / '1' / 'ses-01'。"""
    if isinstance(session, str) and session.startswith("ses-"):
        return session
    return f"ses-{int(session):02d}"


def raw_session_eeg_dir(subject, session, root=None) -> Path:
    """原始 BDF 所在目录：<root>/sourcedata/2C dataset/sub-XXX/ses-YY/eeg。"""
    return dataset_root(root) / RAW_SUBDIR / sub_id(subject) / ses_id(session) / "eeg"


def raw_bdf_paths(subject, session, root=None) -> tuple[Path, Path]:
    """返回 (data.bdf, evt.bdf) 路径元组。"""
    d = raw_session_eeg_dir(subject, session, root)
    return d / "data.bdf", d / "evt.bdf"


def derivatives_mat_path(subject, session, root=None) -> Path:
    """论文已处理 .mat 路径（用于 sanity 对照，只读）。"""
    s, ss = sub_id(subject), ses_id(session)
    return (
        dataset_root(root) / DERIV_SUBDIR / s / ss / "eeg"
        / f"{s}_{ss}_task-motorimagery_eeg.mat"
    )


def list_subjects(root=None) -> List[str]:
    """从磁盘枚举所有 sub-XXX（排序）。绝不写死数量。"""
    base = dataset_root(root) / RAW_SUBDIR
    pat = re.compile(r"^sub-\d{3}$")
    return sorted(p.name for p in base.iterdir() if p.is_dir() and pat.match(p.name))


def list_sessions(subject, root=None) -> List[str]:
    """枚举某被试的所有 ses-YY（排序）。"""
    base = dataset_root(root) / RAW_SUBDIR / sub_id(subject)
    pat = re.compile(r"^ses-\d{2}$")
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir() and pat.match(p.name))


def iter_sessions(root=None):
    """生成 (sub-XXX, ses-YY) 的全部组合，按被试/ session 排序。"""
    for s in list_subjects(root):
        for ss in list_sessions(s, root):
            yield s, ss


# --- 项目内部（可写）目录 ---

def project_path(*parts: str) -> Path:
    """相对项目根目录拼出绝对路径。"""
    return PROJECT_ROOT.joinpath(*parts)


def processed_session_dir(variant: str, subject, session) -> Path:
    """处理后数据的输出目录：data/<variant>/sub-XXX/ses-YY。

    variant 例如 'processed_paper_style' 或 'processed_eog_ecg_clean'。
    """
    return project_path("data", variant, sub_id(subject), ses_id(session))
