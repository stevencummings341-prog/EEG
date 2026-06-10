"""路径工具：所有文件系统路径都从 configs/paths.yaml（或环境变量）读取。

设计原则（见 .cursor/rules/10-data-paths）：
  - 原始数据在项目外部，路径绝不写死在 Python 里，只能来自 configs/paths.yaml
    或环境变量 SHU_2C_ROOT。
  - 路径未知/不存在时不猜，抛清晰错误，提示用户填 configs/paths.yaml。
  - 处理后数据写到 configs/paths.yaml 指定目录（默认 outputs/processed_*），
    绝不写入外部原始数据目录。
  - 被试数量从磁盘枚举，绝不写死。
  - 数据集文件夹名 "2C dataset" 含空格，必须用 pathlib 拼接。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

import yaml

# 本文件位于 <root>/src/utils/paths.py，上溯三级得项目根。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATHS_CONFIG = PROJECT_ROOT / "configs" / "paths.yaml"
RAW_ROOT_ENV = "SHU_2C_ROOT"            # 环境变量可覆盖 raw_data.shu_2c_root
_PLACEHOLDER_PREFIX = "/absolute/path/to"


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


def project_path(*parts: str) -> Path:
    """相对项目根目录拼出绝对路径。"""
    return PROJECT_ROOT.joinpath(*parts)


def _resolve(p: str | Path) -> Path:
    """相对路径按项目根解析；绝对路径（如 scratch 目录）原样使用。"""
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


@dataclass
class Paths:
    """集中管理本项目用到的所有路径（由 load_paths 构造）。"""

    raw_root: Path
    raw_subdir: str
    deriv_subdir: str
    paper_style_root: Path
    eog_ecg_clean_root: Path
    raw_manifest: Path
    processed_manifest: Path
    splits_dir: Path

    # --- 原始数据（只读）---
    def raw_session_eeg_dir(self, subject, session) -> Path:
        return self.raw_root / self.raw_subdir / sub_id(subject) / ses_id(session) / "eeg"

    def raw_bdf_paths(self, subject, session) -> Tuple[Path, Path]:
        """返回 (data.bdf, evt.bdf)。"""
        d = self.raw_session_eeg_dir(subject, session)
        return d / "data.bdf", d / "evt.bdf"

    def derivatives_mat_path(self, subject, session) -> Path:
        """论文已处理 .mat 路径（仅作标签对照真值，非主训练入口）。"""
        s, ss = sub_id(subject), ses_id(session)
        return (self.raw_root / self.deriv_subdir / s / ss / "eeg"
                / f"{s}_{ss}_task-motorimagery_eeg.mat")

    def list_subjects(self) -> List[str]:
        """从磁盘枚举 sub-XXX（排序）。绝不写死数量。"""
        base = self.raw_root / self.raw_subdir
        pat = re.compile(r"^sub-\d{3}$")
        if not base.exists():
            return []
        return sorted(p.name for p in base.iterdir() if p.is_dir() and pat.match(p.name))

    def list_sessions(self, subject) -> List[str]:
        base = self.raw_root / self.raw_subdir / sub_id(subject)
        pat = re.compile(r"^ses-\d{2}$")
        if not base.exists():
            return []
        return sorted(p.name for p in base.iterdir() if p.is_dir() and pat.match(p.name))

    def iter_sessions(self) -> Iterator[Tuple[str, str]]:
        for s in self.list_subjects():
            for ss in self.list_sessions(s):
                yield s, ss

    # --- 处理后数据（可写）---
    def processed_dir(self, variant: str) -> Path:
        """按模式/变体返回处理后数据根目录。"""
        if variant == "paper_style":
            return self.paper_style_root
        if variant == "eog_ecg_clean":
            return self.eog_ecg_clean_root
        # 其他变体：在 paper_style 同级新建
        return self.paper_style_root.parent / f"processed_{variant}"

    def processed_session_dir(self, variant: str, subject, session) -> Path:
        return self.processed_dir(variant) / sub_id(subject) / ses_id(session)

    def session_npz_path(self, variant: str, subject, session) -> Path:
        """正式输出的 per-session .npz 路径（命名遵循 BIDS-like 习惯）。"""
        s, ss = sub_id(subject), ses_id(session)
        return self.processed_session_dir(variant, s, ss) / f"{s}_{ss}_task-motorimagery_eeg.npz"


def load_paths(config_path: str | Path | None = None, require_raw: bool = True) -> Paths:
    """读取 configs/paths.yaml，构造并校验 Paths。

    require_raw=True 时校验原始数据根存在（用于真正要读 raw 的脚本）；
    生成 manifest 之外、仅需输出路径的场景可设 False。
    """
    cfg_path = Path(config_path) if config_path else DEFAULT_PATHS_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"未找到路径配置 {cfg_path}。请创建 configs/paths.yaml 并填写 raw_data.shu_2c_root。"
        )
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    raw = cfg.get("raw_data", {}) or {}
    proc = cfg.get("processed_data", {}) or {}
    man = cfg.get("manifests", {}) or {}
    spl = cfg.get("splits", {}) or {}

    raw_root_str = os.environ.get(RAW_ROOT_ENV) or raw.get("shu_2c_root")
    if not raw_root_str or str(raw_root_str).startswith(_PLACEHOLDER_PREFIX):
        raise ValueError(
            "raw_data.shu_2c_root 未设置或仍是占位符。请在 configs/paths.yaml 填写真实外部"
            f" 路径，或设置环境变量 {RAW_ROOT_ENV}。"
        )
    raw_root = Path(raw_root_str)
    if require_raw and not raw_root.exists():
        raise FileNotFoundError(
            f"原始数据根不存在: {raw_root}。请修正 configs/paths.yaml 中的 raw_data.shu_2c_root。"
        )

    # 处理后输出键：优先用新的 *_root；向后兼容旧的 *_dir 键。
    paper_style_root = proc.get("paper_style_root", proc.get("paper_style_dir",
                                                             "outputs/processed_paper_style"))
    eog_ecg_clean_root = proc.get("eog_ecg_clean_root", proc.get("eog_ecg_clean_dir",
                                                                 "outputs/processed_eog_ecg_clean"))

    return Paths(
        raw_root=raw_root,
        raw_subdir=raw.get("raw_subdir", "sourcedata/2C dataset"),
        deriv_subdir=raw.get("derivatives_subdir", "derivatives/2C dataset_processeddata"),
        paper_style_root=_resolve(paper_style_root),
        eog_ecg_clean_root=_resolve(eog_ecg_clean_root),
        raw_manifest=_resolve(man.get("raw_manifest", "manifests/shu_2c_raw_manifest.csv")),
        processed_manifest=_resolve(man.get("processed_manifest", "manifests/shu_2c_processed_manifest.csv")),
        splits_dir=_resolve(spl.get("dir", "splits")),
    )


def assert_safe_output_dir(out_dir: str | Path, P: "Paths") -> Path:
    """确保 out_dir 是安全的写入位置：绝不写入 raw 的 sourcedata/derivatives/code，
    也不写入 raw 数据根本身。返回解析后的绝对路径；不安全则抛错。

    允许的位置例如 <raw_root>/processed/... 或项目内 outputs/...。
    """
    out = Path(out_dir).resolve()
    raw_root = P.raw_root.resolve()
    forbidden = [
        (raw_root / P.raw_subdir).resolve(),                 # sourcedata/2C dataset
        (raw_root / P.deriv_subdir).resolve(),               # derivatives/...
        (raw_root / "sourcedata").resolve(),
        (raw_root / "derivatives").resolve(),
        (raw_root / "code").resolve(),
    ]
    for fb in forbidden:
        # out 等于禁区、在禁区之内、或是禁区的祖先（例如直接写 raw_root）都拒绝。
        if out == fb or fb in out.parents or out in fb.parents:
            raise ValueError(
                f"拒绝写入受保护的原始数据区域：out={out} 与禁区 {fb} 冲突。"
                " 正式预处理只能写入 processed/ 子目录（见 configs/paths.yaml）。"
            )
    if out == raw_root:
        raise ValueError(f"拒绝直接写入原始数据根 {raw_root}。")
    return out


def ensure_writable_dir(path: str | Path) -> Path:
    """目录不存在则创建；随后做一次写测试。不可写直接抛错（不继续）。"""
    p = Path(path)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise PermissionError(f"无法创建输出目录 {p}: {e}") from e
    probe = p / f".write_test_{os.getpid()}"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as e:
        raise PermissionError(f"输出目录不可写: {p} ({e})") from e
    return p
