"""路径工具：所有文件系统路径都从 configs/paths*.yaml（或环境变量）读取。

设计原则（见 .cursor/rules/10-data-paths）：
  - 原始数据在项目外部，路径绝不写死在 Python 里，只能来自路径配置
    或环境变量（SHU_2C_ROOT / SHU_ROOT 等）。
  - 本机覆盖优先：`paths.local.yaml`（已 gitignore）> `paths.yaml`。
  - 路径未知/不存在时不猜，抛清晰错误，提示用户从 example 复制并填写。
  - 处理后数据写到配置指定目录（默认 outputs/processed_*），
    绝不写入外部原始数据目录。
  - 被试数量从磁盘枚举，绝不写死。
  - 数据集文件夹名 "2C dataset" 含空格，必须用 pathlib 拼接。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import yaml

def _find_project_root() -> Path:
    """Find the project root from either legacy src/ or the new code/ tree."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "AGENTS.md").exists() and (parent / ".git").exists():
            return parent
    # Fallback for copied code/utils/paths.py: <root>/code/utils/paths.py
    return here.parents[2]


PROJECT_ROOT = _find_project_root()
DEFAULT_PATHS_CONFIG = PROJECT_ROOT / "code" / "configs" / "paths.yaml"
LOCAL_PATHS_CONFIG = PROJECT_ROOT / "code" / "configs" / "paths.local.yaml"
RAW_ROOT_ENV = "SHU_2C_ROOT"            # 覆盖 WBCIC / SHU-2C raw 根
SHU_ROOT_ENV = "SHU_ROOT"               # 覆盖 SHU 2022 raw 根
SHU_MANIFEST_ENV = "SHU_PROCESSED_MANIFEST"
_PLACEHOLDER_MARKERS = ("/CHANGE/ME", "/absolute/path/to", "CHANGE/ME")


def prefer_local_config(default: Path, local_name: Optional[str] = None) -> Path:
    """若同目录存在 `*.local.yaml`，优先用之（跨机本机路径，不进 git）。"""
    local = default.with_name(
        local_name or default.name.replace(".yaml", ".local.yaml").replace(".yml", ".local.yml")
    )
    if local.exists():
        return local
    return default


def resolve_config_path(default_rel: str | Path) -> Path:
    """相对项目根的配置路径，自动优先 *.local.yaml。"""
    default = Path(default_rel)
    if not default.is_absolute():
        default = PROJECT_ROOT / default
    return prefer_local_config(default)


def _is_placeholder(value: object) -> bool:
    s = str(value or "")
    return (not s) or any(m in s for m in _PLACEHOLDER_MARKERS)


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
    shu_raw_root: Path
    shu_npz_clean_root: Path
    shu_processed_manifest: Path
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
    """读取路径配置，构造并校验 Paths。

    默认优先 `code/configs/paths.local.yaml`，否则 `paths.yaml`。
    require_raw=True 时校验 WBCIC raw 根存在（真正读 raw 时）；
    仅需 processed / manifest 的场景可设 False。
    """
    if config_path is None:
        cfg_path = prefer_local_config(DEFAULT_PATHS_CONFIG)
        if not cfg_path.exists() and LOCAL_PATHS_CONFIG.exists():
            cfg_path = LOCAL_PATHS_CONFIG
    else:
        cfg_path = prefer_local_config(Path(config_path))
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"未找到路径配置 {cfg_path}。请执行: "
            "cp code/configs/paths.example.yaml code/configs/paths.local.yaml "
            "并填写本机数据路径。"
        )
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    raw = cfg.get("raw_data", {}) or {}
    proc = cfg.get("processed_data", {}) or {}
    man = cfg.get("manifests", {}) or {}
    spl = cfg.get("splits", {}) or {}
    shu = cfg.get("shu", {}) or {}

    raw_root_str = os.environ.get(RAW_ROOT_ENV) or raw.get("shu_2c_root")
    if _is_placeholder(raw_root_str):
        raise ValueError(
            "raw_data.shu_2c_root 未设置或仍是占位符。请在 paths.local.yaml 填写真实外部"
            f" 路径，或设置环境变量 {RAW_ROOT_ENV}。"
        )
    raw_root = Path(str(raw_root_str))
    if require_raw and not raw_root.exists():
        raise FileNotFoundError(
            f"原始数据根不存在: {raw_root}。请修正 paths.local.yaml 中的 raw_data.shu_2c_root。"
        )

    # 处理后输出键：优先用新的 *_root；向后兼容旧的 *_dir 键。
    paper_style_root = proc.get("paper_style_root", proc.get("paper_style_dir",
                                                             "outputs/processed_paper_style"))
    eog_ecg_clean_root = proc.get("eog_ecg_clean_root", proc.get("eog_ecg_clean_dir",
                                                                 "outputs/processed_eog_ecg_clean"))

    shu_raw = os.environ.get(SHU_ROOT_ENV) or shu.get("raw_root") or shu.get("data_dir")
    shu_npz = (
        shu.get("npz_clean_root")
        or shu.get("processed_root")
        or proc.get("shu_npz_clean_root")
    )
    shu_manifest = (
        os.environ.get(SHU_MANIFEST_ENV)
        or man.get("shu_processed_manifest")
        or shu.get("manifest")
    )
    # Soft defaults: allow WBCIC-only machines to omit SHU until needed.
    if _is_placeholder(shu_raw):
        shu_raw = "outputs/external_missing/SHU"
    if _is_placeholder(shu_npz):
        shu_npz = "outputs/external_missing/SHU/processed/npz_clean"
    if _is_placeholder(shu_manifest):
        shu_manifest = "outputs/external_missing/SHU/processed/npz_clean/processed_manifest.csv"

    return Paths(
        raw_root=raw_root,
        raw_subdir=raw.get("raw_subdir", "sourcedata/2C dataset"),
        deriv_subdir=raw.get("derivatives_subdir", "derivatives/2C dataset_processeddata"),
        paper_style_root=_resolve(paper_style_root),
        eog_ecg_clean_root=_resolve(eog_ecg_clean_root),
        raw_manifest=_resolve(man.get("raw_manifest", "manifests/shu_2c_raw_manifest.csv")),
        processed_manifest=_resolve(man.get("processed_manifest", "manifests/shu_2c_processed_manifest.csv")),
        shu_raw_root=_resolve(shu_raw),
        shu_npz_clean_root=_resolve(shu_npz),
        shu_processed_manifest=_resolve(shu_manifest),
        splits_dir=_resolve(spl.get("dir", "splits")),
    )


def resolve_manifest_path(cfg: Dict[str, Any], P: Optional[Paths] = None) -> Path:
    """从实验 config 解析 processed manifest（逻辑键或真实路径）。

    支持:
      - data.manifest = "processed_manifest"           -> WBCIC (paths.yaml)
      - data.manifest = "shu_processed_manifest"       -> SHU
      - data.name = "shu" 且未给文件路径              -> SHU
      - 含 `/` 或以 `.csv` 结尾的字符串               -> 绝对/相对项目根路径
    """
    if P is None:
        P = load_paths(require_raw=False)
    data = cfg.get("data") or {}
    m = data.get("manifest") or data.get("manifest_path")
    name = str(data.get("name") or cfg.get("dataset") or "").lower()

    if m in (None, "", "processed_manifest"):
        if name in ("shu", "shu_2022"):
            return P.shu_processed_manifest
        return P.processed_manifest
    if m in ("shu_processed_manifest", "shu_manifest"):
        return P.shu_processed_manifest

    m_str = str(m)
    if "/" in m_str or m_str.endswith(".csv"):
        return _resolve(m_str)
    # Unknown logical key: fail clearly rather than silently using WBCIC.
    raise ValueError(
        f"未知 data.manifest={m!r}。请用 processed_manifest / shu_processed_manifest，"
        "或填写指向 processed_manifest.csv 的路径。"
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
