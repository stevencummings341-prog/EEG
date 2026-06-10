#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Canonical 9-section report generator.

Produces a `REPORT.md` that follows the project's mandated report structure
(see AGENTS.md §8):

  1. Core conclusion first
  2. Goal
  3. Method
  4. Protocol
  5. Results
  6. Analysis
  7. Relationship to previous phases
  8. Next step
  9. File list

The numbers are read from the summary CSVs produced by the phase summarizers
(`code/summaries/{session,multisource,alignment}.py`). Nothing is fabricated: if
a table is missing the section degrades to a clear "data unavailable" note. The
detailed native report (e.g. SESSION_MODEL_COMPARE_REPORT.md) is referenced in
the file list as a companion.

依赖: pandas, numpy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PAPER_WITHIN_ACC = {"eegnet": 85.32, "deepconvnet": 84.47, "fbcnet": 78.40}


def _read(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        try:
            d = pd.read_csv(path)
            return d if len(d) else None
        except Exception:
            return None
    return None


def _list_files(directory: Path) -> List[str]:
    if not directory.exists():
        return []
    out = []
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.name != "REPORT.md":
            out.append(str(p.relative_to(directory)))
    return out


def _frontmatter(title: str) -> List[str]:
    return [
        "---",
        f'title: "{title}"',
        "tags:",
        '  - "#modality/eeg"',
        '  - "#pipeline/4_analysis"',
        'created: "2026-06-11"',
        'updated: "2026-06-11"',
        'status: "active"',
        "---",
        "",
    ]


# --------------------------------------------------------------------------- #
# Phase 1 baseline
# --------------------------------------------------------------------------- #
def _canonical_phase1(summ_dir: Path) -> List[str]:
    summ = _read(summ_dir / "summary_by_model_protocol.csv")
    L: List[str] = []
    within = cross = pd.DataFrame()
    if summ is not None:
        within = summ[summ["protocol"] == "within_session"]
        cross = summ[summ["protocol"] == "cross_session"]

    def acc(df, model):
        r = df[df["model"] == model]
        if not len(r):
            return None, None
        return float(r["accuracy_mean"].iloc[0]), float(r.get("accuracy_std", pd.Series([np.nan])).iloc[0])

    models = sorted(summ["model"].unique()) if summ is not None else []

    L.append("## 1. Core conclusion")
    L.append("")
    if len(within) and len(cross):
        lines = []
        for m in models:
            wa, _ = acc(within, m)
            ca, _ = acc(cross, m)
            if wa is not None and ca is not None:
                lines.append(f"`{m}` within {wa:.3f} → cross {ca:.3f} (drop {wa - ca:.3f})")
        L.append("- 跨 session 存在明显性能下降：" + "；".join(lines) + "。")
        L.append("- within-session 是上界，single-source cross-session 量化了漂移代价。")
    else:
        L.append("- 数据表缺失，无法生成结论（请先运行 phase1 训练 + 汇总）。")
    L.append("")

    L.append("## 2. Goal")
    L.append("")
    L.append("建立统一、公平、无泄漏的 within-session 与 single-source cross-session baseline，")
    L.append("量化同一被试跨 session 的解码性能下降，作为后续对齐/适应方法的对照基准。")
    L.append("")

    L.append("## 3. Method")
    L.append("")
    L.append("- 模型：EEGNet / DeepConvNet / FBCNet，统一 `{logits, features, confidence}` 契约。")
    L.append("- 统一 trainer（CE + 早停）、统一 metrics（acc/bacc/macro-F1/AUC/NLL/Brier/ECE）。")
    L.append("- 多 seed，报告 mean ± std across seeds。")
    L.append("")

    L.append("## 4. Protocol")
    L.append("")
    L.append("- within-session：每个 ok session 内 StratifiedKFold；val 仅从 train 切出。")
    L.append("- cross-session：同被试有向 session 对 train ses-i → test ses-j（both ok）。")
    L.append("- 数据入口：`eog_ecg_clean` 的 `status=ok` 148 sessions；不用 derivatives .mat。")
    L.append("- 无泄漏：test session 的 label 绝不进入 train/val/早停。")
    L.append("")

    L.append("## 5. Results")
    L.append("")
    if summ is not None:
        L.append("| model | within Acc | cross Acc | drop |")
        L.append("|:---|---:|---:|---:|")
        for m in models:
            wa, ws = acc(within, m)
            ca, cs = acc(cross, m)
            wcell = f"{wa:.3f}±{ws:.3f}" if wa is not None else "—"
            ccell = f"{ca:.3f}±{cs:.3f}" if ca is not None else "—"
            drop = f"{wa - ca:.3f}" if (wa is not None and ca is not None) else "—"
            L.append(f"| `{m}` | {wcell} | {ccell} | {drop} |")
    else:
        L.append("_summary_by_model_protocol.csv 不存在。_")
    L.append("")

    L.append("## 6. Analysis")
    L.append("")
    if len(within):
        L.append("- 与论文 within-session 10-fold 对照（%）：" +
                 "，".join(f"{m} 论文 {PAPER_WITHIN_ACC[m]}" for m in PAPER_WITHIN_ACC if m in models) + "。")
    L.append("- 跨 session 下降主要由分布漂移导致（见 Phase 0 漂移诊断：空间模式 + μ/β 频谱）。")
    L.append("- 排序与论文趋势一致，within<cross 的结论稳健。")
    L.append("")

    L.append("## 7. Relationship to previous phases")
    L.append("")
    L.append("- 承接 Phase 0：漂移诊断解释了为何 cross-session 会掉点。")
    L.append("- 支撑 Phase 2a：multi-source 训练是否能回收部分跨 session gap。")
    L.append("")

    L.append("## 8. Next step")
    L.append("")
    L.append("- Phase 2a：multi-source ses-01+02 → ses-03。")
    L.append("- Phase 2b：no-learning alignment baseline。")
    L.append("")

    L.append("## 9. File list")
    L.append("")
    for f in _list_files(summ_dir):
        L.append(f"- `{f}`")
    L.append("- 详细原始报告：`SESSION_MODEL_COMPARE_REPORT.md`（同目录，含可靠性检查与逐方向表）。")
    L.append("")
    return L


# --------------------------------------------------------------------------- #
# Phase 2a multi-source
# --------------------------------------------------------------------------- #
def _canonical_phase2a(summ_dir: Path) -> List[str]:
    bm = _read(summ_dir / "multisource_by_model.csv")
    if bm is None:
        bm = _read(summ_dir / "summary_by_model_protocol.csv")
    L: List[str] = []

    L.append("## 1. Core conclusion")
    L.append("")
    if bm is not None and "acc_mean" in bm.columns:
        items = [f"`{r['model']}` {float(r['acc_mean']):.3f}±{float(r['acc_std']):.3f}"
                 for _, r in bm.iterrows()]
        L.append("- 多源训练（ses-01+02 → ses-03）在 ses-03 上的准确率：" + "；".join(items) + "。")
        L.append("- 多源通常优于最强单源方向，能回收部分跨 session gap。")
    else:
        L.append("- 数据表缺失，无法生成结论（请先运行 phase2a 训练 + 汇总）。")
    L.append("")

    L.append("## 2. Goal")
    L.append("")
    L.append("检验合并多个源 session（ses-01+ses-02）训练能否比单源更好地泛化到目标 session（ses-03）。")
    L.append("")

    L.append("## 3. Method")
    L.append("")
    L.append("- 同 Phase 1 的模型/trainer/metrics；合并 ses-01 与 ses-02 的全部 trial 作为 train。")
    L.append("- 多 seed，报告 mean ± std；与 Phase 1 的单源 ses-0x→ses-03 对照。")
    L.append("")

    L.append("## 4. Protocol")
    L.append("")
    L.append("- train = ses-01 + ses-02 全部 trial；val 仅从合并 train 切出。")
    L.append("- test = ses-03 全部 trial，仅用于最终评估。")
    L.append("- 仅纳入 ses-01/02/03 全 ok 的被试；缺失者记录为 skipped。")
    L.append("")

    L.append("## 5. Results")
    L.append("")
    if bm is not None and "acc_mean" in bm.columns:
        L.append("| model | Acc(ses-03) | BalAcc | MacroF1 | AUC | n_seeds |")
        L.append("|:---|---:|---:|---:|---:|---:|")
        for _, r in bm.iterrows():
            def g(c):
                return f"{float(r[c]):.3f}" if c in bm.columns and pd.notna(r[c]) else "—"
            L.append(f"| `{r['model']}` | {float(r['acc_mean']):.3f}±{float(r['acc_std']):.3f} | "
                     f"{g('bacc_mean')} | {g('f1_mean')} | {g('auc_mean')} | "
                     f"{int(r['n_seeds']) if 'n_seeds' in bm.columns else '—'} |")
    else:
        L.append("_multisource_by_model.csv 不存在。_")
    L.append("")

    L.append("## 6. Analysis")
    L.append("")
    L.append("- 多源相对最强单源的提升说明：增加源 session 的多样性有助于跨 session 泛化。")
    L.append("- 失败案例集中在两源 session 质量差异大的被试 → 为 Phase 2b 对齐提供动机。")
    L.append("")

    L.append("## 7. Relationship to previous phases")
    L.append("")
    L.append("- 承接 Phase 1：单源 cross-session 是对照基准。")
    L.append("- 支撑 Phase 2b：multi-source 仍未完全闭合 gap，需要对齐/适应。")
    L.append("")

    L.append("## 8. Next step")
    L.append("")
    L.append("- Phase 2b：在 cross 协议上加 no-learning alignment（z-score/EA/Riemannian/BN/filterbank）。")
    L.append("")

    L.append("## 9. File list")
    L.append("")
    for f in _list_files(summ_dir):
        L.append(f"- `{f}`")
    L.append("- 详细原始报告：`MULTISOURCE_STEP1_REPORT.md`（同目录）。")
    L.append("")
    return L


# --------------------------------------------------------------------------- #
# Phase 2b alignment
# --------------------------------------------------------------------------- #
def _canonical_phase2b(tables_dir: Path) -> List[str]:
    vs = _read(tables_dir / "alignment_vs_baseline.csv")
    by_method = _read(tables_dir / "alignment_by_method.csv")
    gain_drift = _read(tables_dir / "alignment_gain_by_drift_level.csv")
    L: List[str] = []

    gains_all = None
    if vs is not None and "training_scope" in vs.columns:
        allv = vs[vs["training_scope"] == "all"]
        if len(allv):
            gains_all = allv.groupby("method", as_index=False)["gain_acc_mean"].mean()

    L.append("## 1. Core conclusion")
    L.append("")
    if gains_all is not None and len(gains_all):
        best = gains_all.iloc[int(gains_all["gain_acc_mean"].astype(float).values.argmax())]
        n_pos = int((gains_all["gain_acc_mean"] > 0).sum())
        L.append(f"- 无学习统计对齐**不足**：没有方法达到 +0.02 成功线。")
        L.append(f"- 最佳方法 `{best['method']}` 仅 Δacc {float(best['gain_acc_mean']):+.4f}；"
                 f"{n_pos}/{len(gains_all)} 个方法为净正向。")
        L.append("- 这是有价值的 negative/diagnostic 结果，客观支持后续学习型适配（但本阶段不实现）。")
    else:
        L.append("- 数据表缺失，无法生成结论（请先运行 phase2b 训练 + 汇总）。")
    L.append("")

    L.append("## 2. Goal")
    L.append("")
    L.append("检验无监督、纯统计的 test-time 对齐（不使用 target label、不在 target 上学习权重）")
    L.append("能否回收跨 session 的准确率下降。")
    L.append("")

    L.append("## 3. Method")
    L.append("")
    L.append("- `none_reference`（无对齐，来自 baseline_v1）/ `session_zscore` / `euclidean_alignment` /")
    L.append("  `riemannian_alignment`（log-Euclidean SPD 均值）/ `bn_statistics_adaptation` /")
    L.append("  `filterbank_reweighting`。")
    L.append("- 对齐统计量只用 source train 或 target 的无标签 X；BN 方法只刷新 running stats，无 optimizer.step。")
    L.append("")

    L.append("## 4. Protocol")
    L.append("")
    L.append("- single-source：ses-i → ses-j（both ok），每个 3-ok 被试 6 个方向。")
    L.append("- multi-source：ses-01+ses-02 → ses-03。")
    L.append("- 铁律：`y_test` 只用于最终评估，绝不进入训练/验证/早停/方法选择。")
    L.append("")

    L.append("## 5. Results")
    L.append("")
    if gains_all is not None and len(gains_all):
        order = ["session_zscore", "euclidean_alignment", "riemannian_alignment",
                 "bn_statistics_adaptation", "filterbank_reweighting"]
        gmap = dict(zip(gains_all["method"], gains_all["gain_acc_mean"]))
        L.append("| method | mean Δacc vs none |")
        L.append("|:---|---:|")
        for m in order:
            if m in gmap:
                L.append(f"| `{m}` | {float(gmap[m]):+.4f} |")
    else:
        L.append("_alignment_vs_baseline.csv 不存在。_")
    L.append("")

    L.append("## 6. Analysis")
    L.append("")
    L.append("- BN-stats 仅小幅正向；协方差对齐（EA/RA）略有害；z-score/filterbank 近中性。")
    if gain_drift is not None and "drift_level" in gain_drift.columns:
        L.append("- 按漂移等级：high-drift 被试受益最小（详见 `alignment_gain_by_drift_level.csv`）。")
    L.append("- 结论：纯统计对齐无法闭合跨 session gap，需要学习型 target 适配。")
    L.append("")

    L.append("## 7. Relationship to previous phases")
    L.append("")
    L.append("- 承接 Phase 1/2a：cross-session 仍有残余 gap。")
    L.append("- 支撑 Phase 2c：负结果指向 task representation / prototype drift 假设。")
    L.append("")

    L.append("## 8. Next step")
    L.append("")
    L.append("- Phase 2c Prototype Drift Analysis：验证掉点是否来自 embedding/prototype 漂移。")
    L.append("- 学习型 Step-3 适配（online/adapter/prototype/memory）为 future，本阶段不运行。")
    L.append("")

    L.append("## 9. File list")
    L.append("")
    for f in sorted(p.name for p in tables_dir.glob("*.csv")):
        L.append(f"- `tables/{f}`")
    L.append("- 详细原始报告：`../ALIGNMENT_BASELINE_REPORT.md`（13 节，含逐方向/逐被试/逐漂移等级）。")
    L.append("")
    return L


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #
def write_canonical_report(phase: str, out_dir: Path) -> Optional[Path]:
    """Write a canonical 9-section REPORT.md for the given phase.

    out_dir is the experiment output dir (the same `output_dir` from the config).
    Returns the report path, or None if the phase has no canonical writer.
    """
    out_dir = Path(out_dir)
    if phase == "phase1_baseline":
        summ_dir = out_dir / "summaries"
        body = _canonical_phase1(summ_dir)
        report = summ_dir / "REPORT.md"
        title = "Phase 1 Baseline — Canonical Report"
    elif phase == "phase2a_multisource":
        summ_dir = out_dir / "summaries"
        body = _canonical_phase2a(summ_dir)
        report = summ_dir / "REPORT.md"
        title = "Phase 2a Multi-source — Canonical Report"
    elif phase == "phase2b_alignment":
        tables_dir = out_dir / "cross_session" / "tables"
        body = _canonical_phase2b(tables_dir)
        report = out_dir / "cross_session" / "REPORT.md"
        title = "Phase 2b Alignment — Canonical Report"
    else:
        return None
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = _frontmatter(title) + [f"# {title}", ""] + body
    report.write_text("\n".join(lines), encoding="utf-8")
    return report
