"""读取 Neuracle evt.bdf 中的事件（绕过 MNE 解析不全的问题）。

背景（见 docs/DATASET_SHU.md「Events」与 PROGRESS）：
  SHU 数据集的 evt.bdf 是 BDF+C 文件，事件存在 "BDF Annotations"(TAL) 通道里。
  MNE 的 read_raw_bdf 只能解出 2 条（块标记 7/8），漏掉了 200 个试次触发。
  本模块直接解析 BDF 头 + TAL 文本，提取全部 (onset_seconds, code) 事件。

已在 sub-001/ses-01 验证：得到 100 个 '1'(左手) + 100 个 '2'(右手)，间隔约 8s。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# TAL（Timestamped Annotation List）分隔符。
_TAL_SEP = "\x14"      # 0x14 分隔 onset/duration 与 description，并结束一条描述
_TAL_DUR = "\x15"      # 0x15 分隔 onset 与 duration
_TAL_END = "\x00"      # 0x00 结束一个 TAL 块


@dataclass
class BdfHeader:
    n_records: int
    record_duration: float           # 每个 data record 的时长（秒）
    n_signals: int
    labels: List[str]
    n_samples_per_record: List[int]  # 每个信号在一个 record 内的采样点数
    data_offset: int                 # 数据区起始字节偏移


def parse_bdf_header(path: str | Path) -> BdfHeader:
    """解析 BDF/EDF 头部（通用，不写死信号数量）。"""
    with open(path, "rb") as f:
        head = f.read(256)
    if head[0] != 255:
        raise ValueError(f"不是 BDF 文件（首字节应为 255）: {path}")

    def fld(buf: bytes, a: int, z: int) -> str:
        return buf[a:z].decode("latin-1").strip()

    n_records = int(fld(head, 236, 244))
    record_duration = float(fld(head, 244, 252))
    ns = int(fld(head, 252, 256))

    with open(path, "rb") as f:
        sig_head = f.read(256 + 256 * ns)[256:]

    def sfld(i: int, base: int, width: int) -> str:
        a = base + i * width
        return sig_head[a:a + width].decode("latin-1").strip()

    labels = [sfld(i, 0, 16) for i in range(ns)]
    # 跳过 transducer(80) phys_dim(8) pmin(8) pmax(8) dmin(8) dmax(8) prefilter(80)
    nsamp_base = ns * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80)
    n_samples = [int(sfld(i, nsamp_base, 8)) for i in range(ns)]

    data_offset = 256 * (ns + 1)
    return BdfHeader(n_records, record_duration, ns, labels, n_samples, data_offset)


def _annotation_signal_index(hdr: BdfHeader) -> int:
    """找到注释(TAL)信号的下标（标签里含 'Annotations'）。"""
    for i, lab in enumerate(hdr.labels):
        if "annotation" in lab.lower():
            return i
    raise ValueError(f"未找到 BDF Annotations 通道，labels={hdr.labels}")


def read_neuracle_tals(path: str | Path) -> List[Tuple[float, str]]:
    """解析 evt.bdf，返回全部 (onset_seconds, description) 注释（含块标记）。"""
    hdr = parse_bdf_header(path)
    ann_idx = _annotation_signal_index(hdr)

    # BDF 每个采样点 3 字节；每个信号每 record 占 nsamp*3 字节。
    bytes_per_sample = 3
    sizes = [n * bytes_per_sample for n in hdr.n_samples_per_record]
    rec_size = sum(sizes)
    pre = sum(sizes[:ann_idx])          # 注释信号前面的字节数（每 record）
    ann_len = sizes[ann_idx]            # 注释信号每 record 的字节数

    with open(path, "rb") as f:
        raw = f.read()

    ann_bytes = bytearray()
    for rec in range(hdr.n_records):
        base = hdr.data_offset + rec * rec_size + pre
        ann_bytes += raw[base:base + ann_len]

    text = bytes(ann_bytes).decode("latin-1")
    events: List[Tuple[float, str]] = []
    for block in text.split(_TAL_END):
        if not block or _TAL_SEP not in block:
            continue
        parts = block.split(_TAL_SEP)
        onset_field = parts[0].split(_TAL_DUR)[0]  # 去掉可能的 duration
        # onset_field 可能带前导 '+' 或非数字（record 时间keeping），尝试转 float
        try:
            onset = float(onset_field)
        except ValueError:
            continue
        for desc in parts[1:]:
            if desc != "":
                events.append((onset, desc))
    return events


def read_mi_events(
    path: str | Path,
    code_to_label: Dict[int, int] | None = None,
) -> Tuple[List[float], List[int]]:
    """只返回 MI 试次事件的 (onsets_seconds, labels)。

    code_to_label: 原始触发码 -> 内部标签，默认 {1:0, 2:1}（左手/右手）。
    返回的 onsets 与 labels 按时间排序。
    """
    if code_to_label is None:
        code_to_label = {1: 0, 2: 1}
    valid_codes = set(code_to_label.keys())

    pairs: List[Tuple[float, int]] = []
    for onset, desc in read_neuracle_tals(path):
        try:
            code = int(desc)
        except ValueError:
            continue
        if code in valid_codes:
            pairs.append((onset, code_to_label[code]))

    pairs.sort(key=lambda p: p[0])
    onsets = [p[0] for p in pairs]
    labels = [p[1] for p in pairs]
    return onsets, labels
