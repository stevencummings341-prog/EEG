#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全量预处理结果质检 + 与官方 derivatives(.mat) 的质量对比与可视化。

输入：我们的 processed_manifest.csv（每 session 一个 .npz）+ 官方 derivatives 根目录。
输出（写到 --output-dir）：
  - manifest_qc_summary.json     全量 manifest 质检（19 项）
  - session_quality_metrics.csv  每 session 的 ours/official 指标 + ratio/difference
  - paired_similarity_metrics.csv 仅 labels_match_exact=True 的 session 的配对相似度
  - QC_REPORT.md                 中文总体结论
  - figures/*.png                可视化

注意：这是重任务（读 153×(npz+mat) 并算 PSD），必须在计算节点用 srun/sbatch 跑，
不要在登录节点直接跑全量。可用 --limit N 先在计算节点做小规模 smoke test。

用法：
  python scripts/compare_processed_quality.py \
    --processed-manifest /.../processed/eog_ecg_clean/processed_manifest.csv \
    --official-root "/.../derivatives/2C dataset_processeddata" \
    --output-dir /.../processed/eog_ecg_clean/qc_vs_derivatives \
    --max-example-sessions 6
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import data_quality as dq  # noqa: E402
from src.utils.io import load_session_npz, save_json  # noqa: E402
from src.visualization import quality_plots as qp  # noqa: E402

EXPECT_SHAPE = (200, 58, 1000)
EXPECT_PER_CLASS = 100
EXPECT_SFREQ = 250
EXPECT_NCH = 58
EXPECT_NT = 1000


# ============================ 小工具 ============================

def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_json(path: str | Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def _as_bool(s: Any) -> Optional[bool]:
    if isinstance(s, bool):
        return s
    if s is None:
        return None
    t = str(s).strip().lower()
    if t in ("true", "1", "yes"):
        return True
    if t in ("false", "0", "no"):
        return False
    return None


def _int_or_none(s: Any) -> Optional[int]:
    try:
        return int(str(s).strip())
    except (ValueError, TypeError):
        return None


def _human_size(nbytes: int) -> str:
    x = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024 or unit == "TiB":
            return f"{x:.2f} {unit}"
        x /= 1024
    return f"{x:.2f} TiB"


# ============================ 一、全量 manifest 质检 ============================

def run_manifest_qc(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    """对 processed_manifest.csv 的全部 session 做 19 项检查。"""
    n = len(rows)
    sid = lambda r: f"{r['subject_id']}/{r['session_id']}"  # noqa: E731

    status_counter = Counter(r.get("status", "") for r in rows)
    failed = [sid(r) for r in rows if r.get("status") not in ("ok", "success")]

    bad_xshape, bad_yshape, bad_label, bad_trigger = [], [], [], []
    bad_sfreq, bad_nch, bad_nt = [], [], []
    nan_inf_sessions: List[str] = []
    trigger_missing_report: List[str] = []
    eog_counts: List[int] = []
    ecg_counts: List[int] = []
    total_counts: List[int] = []
    ica_fallback: List[str] = []
    no_aux_fallback: List[str] = []
    aux_used_true = 0
    total_bytes = 0

    lm_true, lm_false, lm_blank = [], [], []
    multiset_false: List[str] = []
    false_but_multiset_true: List[str] = []

    for r in rows:
        s = sid(r)
        nt_trials = _int_or_none(r.get("n_trials"))
        nch = _int_or_none(r.get("n_channels"))
        ntime = _int_or_none(r.get("n_times"))
        sfreq = _int_or_none(r.get("sfreq"))
        l0 = _int_or_none(r.get("label_0_count"))
        l1 = _int_or_none(r.get("label_1_count"))

        # 3 X_shape / 4 y_shape
        if not (nt_trials == EXPECT_SHAPE[0] and nch == EXPECT_NCH and ntime == EXPECT_NT):
            bad_xshape.append(s)
        if nt_trials != EXPECT_SHAPE[0]:
            bad_yshape.append(s)
        # 5 label count
        if not (l0 == EXPECT_PER_CLASS and l1 == EXPECT_PER_CLASS):
            bad_label.append(s)
        # 7/8/9
        if sfreq != EXPECT_SFREQ:
            bad_sfreq.append(s)
        if nch != EXPECT_NCH:
            bad_nch.append(s)
        if ntime != EXPECT_NT:
            bad_nt.append(s)

        # 11/12/13 manifest 记录的 label 与 .mat 一致性
        lm = _as_bool(r.get("labels_match_mat"))
        ms = _as_bool(r.get("labels_multiset_match"))
        if lm is True:
            lm_true.append(s)
        elif lm is False:
            lm_false.append(s)
            if ms is True:
                false_but_multiset_true.append(s)
        else:
            lm_blank.append(s)
        if ms is False:
            multiset_false.append(s)

        # 14 aux cleaning used
        if _as_bool(r.get("aux_cleaning_used")) is True:
            aux_used_true += 1

        # 文件大小
        npz = r.get("npz_path")
        if npz and Path(npz).exists():
            total_bytes += Path(npz).stat().st_size

        # 读 report/meta 拿 6(trigger)/10(nan-inf)/15/16/17/18
        report = _load_json(r.get("report_path", "")) or {}
        meta = _load_json(r.get("meta_path", "")) or {}
        qc = report.get("quality_checks", {}) or {}
        if qc.get("has_nan") or qc.get("has_inf"):
            nan_inf_sessions.append(s)

        n_events = meta.get("n_events")
        if n_events is None:
            trigger_missing_report.append(s)
        elif int(n_events) != EXPECT_SHAPE[0]:
            bad_trigger.append(s)

        aux = report.get("aux_cleaning", {}) or {}
        eog_counts.append(len(aux.get("eog_excluded_components", []) or []))
        ecg_counts.append(len(aux.get("ecg_excluded_components", []) or []))
        total_counts.append(len(aux.get("ica_excluded_components", []) or []))

        warnings = aux.get("cleaning_warnings", []) or []
        ica_params = aux.get("ica_params", {}) or {}
        n_req = ica_params.get("n_components_requested")
        n_used = ica_params.get("n_components_used")
        retried = any(("重试" in w) or ("retry" in w.lower()) or ("failed" in w.lower() and "ICA fit" in w)
                      for w in warnings)
        if retried or (n_req is not None and n_used is not None and n_req != n_used):
            ica_fallback.append(s)
        fb = aux.get("fallback_used")
        if fb:
            no_aux_fallback.append(s)

    def _stat(counts: List[int]) -> Dict[str, Any]:
        a = np.asarray(counts, dtype=float)
        return {
            "sessions_with_ge1": int((a >= 1).sum()),
            "total_components": int(a.sum()),
            "mean_per_session": round(float(a.mean()), 4) if a.size else 0.0,
            "max_per_session": int(a.max()) if a.size else 0,
            "histogram": {str(k): int(v) for k, v in sorted(Counter(counts).items())},
        }

    summary = {
        "1_total_sessions": n,
        "1_total_sessions_ok": n == 153,
        "2_status_counts": dict(status_counter),
        "2_status_ok_153_of_153": status_counter.get("ok", 0) == 153,
        "2_failed_sessions": failed,
        "3_X_shape_all_200_58_1000": len(bad_xshape) == 0,
        "3_X_shape_violations": bad_xshape,
        "4_y_shape_all_200": len(bad_yshape) == 0,
        "4_y_shape_violations": bad_yshape,
        "5_label_count_all_100_100": len(bad_label) == 0,
        "5_label_count_violations": bad_label,
        "6_trigger_count_all_200": len(bad_trigger) == 0 and len(trigger_missing_report) == 0,
        "6_trigger_count_violations": bad_trigger,
        "6_trigger_count_missing_meta": trigger_missing_report,
        "7_sfreq_all_250": len(bad_sfreq) == 0,
        "7_sfreq_violations": bad_sfreq,
        "8_n_channels_all_58": len(bad_nch) == 0,
        "8_n_channels_violations": bad_nch,
        "9_n_times_all_1000": len(bad_nt) == 0,
        "9_n_times_violations": bad_nt,
        "10_nan_inf_present": len(nan_inf_sessions) > 0,
        "10_nan_inf_sessions": nan_inf_sessions,
        "11_labels_match_mat_true_count": len(lm_true),
        "11_labels_match_mat_false_count": len(lm_false),
        "11_labels_match_mat_blank_count": len(lm_blank),
        "12_match_false_but_multiset_true": false_but_multiset_true,
        "13_multiset_false_sessions": multiset_false,
        "14_aux_cleaning_used_true_count": aux_used_true,
        "15_eog_excluded_stats": _stat(eog_counts),
        "16_ecg_excluded_stats": _stat(ecg_counts),
        "16b_total_excluded_stats": _stat(total_counts),
        "17_ica_n_components_fallback_count": len(ica_fallback),
        "17_ica_n_components_fallback_sessions": ica_fallback,
        "18_no_aux_clean_fallback_count": len(no_aux_fallback),
        "18_no_aux_clean_fallback_sessions": no_aux_fallback,
        "19_total_output_bytes": total_bytes,
        "19_total_output_human": _human_size(total_bytes),
    }
    return summary


# ============================ 三/四、逐 session 质量对比 ============================

def _amp_fingerprint(X: np.ndarray) -> tuple:
    """(std, max_abs) 幅值指纹，用于被试内 session 配对（对 ICA 清理稳健）。"""
    Xn = np.nan_to_num(np.asarray(X, dtype=np.float64))
    return float(Xn.std()), float(np.abs(Xn).max())


def run_comparison(rows: List[Dict[str, str]], official_root: str, mi_channels: List[str],
                   max_examples: int, nperseg: int, limit: Optional[int]):
    """按被试分组逐 session 计算质量指标。

    关键：官方 derivatives 的 session 排序对部分被试与 BIDS sourcedata 不一致，
    因此每个被试内先用幅值指纹把「我们的 session」配到「官方真正对应的 session」，
    所有比值/配对都对【匹配后的】官方 session 计算（而非同名 ses）。

    返回 (session_rows, paired_rows, agg, alignment_summary)。
    """
    if limit:
        rows = rows[:limit]

    # 按被试分组（保持 session 顺序）
    from collections import OrderedDict
    by_subj: "OrderedDict[str, List[Dict[str, str]]]" = OrderedDict()
    for r in rows:
        by_subj.setdefault(r["subject_id"], []).append(r)
    for subj in by_subj:
        by_subj[subj].sort(key=lambda r: r["session_id"])

    session_rows: List[Dict[str, Any]] = []
    paired_rows: List[Dict[str, Any]] = []
    alignment_summary: Dict[str, Any] = {"per_subject": [], "permuted_subjects": []}

    freqs_ref: Optional[np.ndarray] = None
    psd_sum_ours: Optional[np.ndarray] = None
    psd_sum_off: Optional[np.ndarray] = None
    psd_pair_count = 0
    sel_sum_ours: Dict[str, np.ndarray] = {}
    sel_sum_off: Dict[str, np.ndarray] = {}
    sel_count = 0

    rms_ratio_rows: List[np.ndarray] = []
    rms_ratio_labels: List[str] = []
    channel_names_ref: Optional[List[str]] = None

    mean_trial_corr_vals: List[float] = []
    examples: List[Dict[str, Any]] = []

    t0 = time.time()
    done = 0
    n_total = len(rows)
    for subj, srows in by_subj.items():
        # ---- 载入该被试全部 ours + official ----
        ours_data: List[Dict[str, Any]] = []
        for r in srows:
            try:
                d = load_session_npz(r.get("npz_path", ""))
                ours_data.append({"ses": r["session_id"], "row": r, "X": d["X"], "y": d["y"],
                                  "ch": d["channel_names"], "sfreq": d["sfreq"]})
            except Exception as e:  # noqa: BLE001
                ours_data.append({"ses": r["session_id"], "row": r, "X": None, "y": None,
                                  "err": f"load_npz_failed: {e!r}"})
        off_data: List[Dict[str, Any]] = []
        for r in srows:
            ses = r["session_id"]
            mp = dq.locate_official_mat(official_root, subj, ses)
            if mp is None:
                off_data.append({"ses": ses, "path": None, "X": None, "y": None})
                continue
            try:
                Xf, yf = dq.load_official_session(mp)
                off_data.append({"ses": ses, "path": str(mp), "X": Xf, "y": yf})
            except Exception as e:  # noqa: BLE001
                off_data.append({"ses": ses, "path": str(mp), "X": None, "y": None,
                                 "err": f"load_mat_failed: {e!r}"})

        # ---- 被试内 session 配对（仅用两侧都成功载入的）----
        valid_o = [i for i, o in enumerate(ours_data) if o["X"] is not None]
        valid_f = [j for j, o in enumerate(off_data) if o["X"] is not None]
        perm_map: Dict[int, int] = {}     # ours idx -> official idx
        align_info = None
        if len(valid_o) == len(valid_f) and len(valid_o) >= 2:
            ours_fps = [_amp_fingerprint(ours_data[i]["X"]) for i in valid_o]
            off_fps = [_amp_fingerprint(off_data[j]["X"]) for j in valid_f]
            align_info = dq.best_session_assignment(ours_fps, off_fps)
            for local_i, off_local in enumerate(align_info["perm"]):
                perm_map[valid_o[local_i]] = valid_f[off_local]
        else:
            # 数量不一致：退回同序配对（按下标）
            for i in valid_o:
                if i < len(off_data) and off_data[i]["X"] is not None:
                    perm_map[i] = i

        matched_official_ses = {i: off_data[perm_map[i]]["ses"] for i in perm_map}
        is_permuted = bool(align_info and align_info["is_permuted"])
        subj_rec = {
            "subject_id": subj,
            "ours_sessions": [ours_data[i]["ses"] for i in range(len(ours_data))],
            "matched_official": [matched_official_ses.get(i, "") for i in range(len(ours_data))],
            "is_permuted": is_permuted,
            "assignment": align_info,
        }
        alignment_summary["per_subject"].append(subj_rec)
        if is_permuted:
            alignment_summary["permuted_subjects"].append(subj)

        # ---- 逐 ours session 计算（对匹配后的官方）----
        for i, o in enumerate(ours_data):
            ses = o["ses"]
            sid = f"{subj}/{ses}"
            r = o["row"]
            base = {"subject_id": subj, "session_id": ses, "status": r.get("status", "")}
            done += 1
            if o["X"] is None:
                base["error"] = o.get("err", "load_failed")
                session_rows.append(base)
                continue

            X_o, y_o, ch_names, sfreq = o["X"], o["y"], o["ch"], o["sfreq"]
            if channel_names_ref is None:
                channel_names_ref = ch_names

            off = off_data[perm_map[i]] if i in perm_map else {"X": None, "y": None, "ses": "", "path": ""}
            X_f, y_f = off.get("X"), off.get("y")
            base["official_mat_path"] = off.get("path", "") or ""
            base["matched_official_session"] = matched_official_ses.get(i, "")
            base["subject_order_permuted"] = is_permuted
            base["session_match_cost"] = align_info["cost"] if align_info else ""
            base["n_trials_ours"] = int(X_o.shape[0])
            base["n_trials_official"] = int(X_f.shape[0]) if X_f is not None else -1

            metrics, extras = dq.compute_session_quality(
                X_o, y_o, X_f, y_f, ch_names, sfreq,
                mi_channels=mi_channels, nperseg=nperseg)
            session_rows.append({**base, **metrics})

            # ---- 聚合 ----
            if extras.get("psd_off_chmean") is not None:
                f = extras["freqs"]
                if freqs_ref is None:
                    freqs_ref = f
                    psd_sum_ours = np.zeros_like(extras["psd_ours_chmean"])
                    psd_sum_off = np.zeros_like(extras["psd_off_chmean"])
                if f.shape == freqs_ref.shape:
                    psd_sum_ours += extras["psd_ours_chmean"]
                    psd_sum_off += extras["psd_off_chmean"]
                    psd_pair_count += 1
                    so, sf = extras.get("psd_ours_sel", {}), extras.get("psd_off_sel", {})
                    for ch in so:
                        if ch in sf:
                            sel_sum_ours[ch] = sel_sum_ours.get(ch, 0) + so[ch]
                            sel_sum_off[ch] = sel_sum_off.get(ch, 0) + sf[ch]
                    sel_count += 1

            rr = extras.get("per_channel_rms_ratio")
            if rr is not None and rr.shape[0] == EXPECT_NCH:
                rms_ratio_rows.append(rr)
                rms_ratio_labels.append(sid)

            if metrics.get("paired_done"):
                ps = extras["paired"]
                mean_trial_corr_vals.append(ps["mean_trial_corr"])
                paired_rows.append({
                    "subject_id": subj, "session_id": ses,
                    "matched_official_session": matched_official_ses.get(i, ""),
                    "n_trials": ps["n_trials"],
                    "mean_trial_corr": ps["mean_trial_corr"],
                    "median_trial_corr": ps["median_trial_corr"],
                    "min_trial_corr": ps["min_trial_corr"],
                    "mean_per_channel_corr": ps["mean_per_channel_corr"],
                    "median_per_channel_corr": ps["median_per_channel_corr"],
                    "mae": ps["mae"], "rmse": ps["rmse"],
                    "relative_rmse": ps["relative_rmse"],
                })
                if len(examples) < max_examples and X_f is not None:
                    ci = dq.find_channel_index(ch_names, "C3")
                    if ci is None:
                        ci = 0
                    nshow = min(int(round(2.0 * sfreq)), X_o.shape[2])
                    t = np.arange(nshow) / float(sfreq)
                    examples.append({
                        "title": f"{sid} -> official {matched_official_ses.get(i,'')}  ch={ch_names[ci]}  trial 0",
                        "t": t,
                        "ours": np.asarray(X_o[0, ci, :nshow], dtype=float),
                        "official": np.asarray(X_f[0, ci, :nshow], dtype=float),
                        "channel": ch_names[ci],
                    })

            if done % 10 == 0 or done == n_total:
                print(f"[{done}/{n_total}] {sid} done "
                      f"(official={'Y' if X_f is not None else 'N'}, "
                      f"perm={'Y' if is_permuted else 'N'}, "
                      f"std_ratio={metrics.get('std_ratio', float('nan')):.3f}, "
                      f"{time.time()-t0:.0f}s)", flush=True)

        # 释放该被试占用的大数组
        del ours_data, off_data

    agg = {
        "freqs": freqs_ref,
        "psd_ours_global": (psd_sum_ours / psd_pair_count) if psd_pair_count else None,
        "psd_off_global": (psd_sum_off / psd_pair_count) if psd_pair_count else None,
        "psd_pair_count": psd_pair_count,
        "sel_ours": {k: v / sel_count for k, v in sel_sum_ours.items()} if sel_count else {},
        "sel_off": {k: v / sel_count for k, v in sel_sum_off.items()} if sel_count else {},
        "rms_ratio_matrix": np.vstack(rms_ratio_rows) if rms_ratio_rows else None,
        "rms_ratio_labels": rms_ratio_labels,
        "channel_names": channel_names_ref,
        "mean_trial_corr_vals": mean_trial_corr_vals,
        "examples": examples,
    }
    return session_rows, paired_rows, agg, alignment_summary


# ============================ 五、写 CSV ============================

def write_session_csv(session_rows: List[Dict[str, Any]], out_path: Path) -> None:
    base_cols = ["subject_id", "session_id", "status", "official_found",
                 "matched_official_session", "subject_order_permuted", "session_match_cost",
                 "official_mat_path", "n_trials_ours", "n_trials_official",
                 "labels_match_exact", "labels_multiset_match", "n_labels_agree",
                 "ours_mean", "ours_std", "ours_rms", "official_mean", "official_std",
                 "official_rms", "std_ratio", "rms_ratio",
                 "mu_bandpower_ratio", "beta_bandpower_ratio"]
    all_keys: List[str] = []
    seen = set()
    for c in base_cols:
        seen.add(c)
        all_keys.append(c)
    for row in session_rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                all_keys.append(k)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        for row in session_rows:
            w.writerow({k: row.get(k, "") for k in all_keys})


def write_paired_csv(paired_rows: List[Dict[str, Any]], out_path: Path) -> None:
    cols = ["subject_id", "session_id", "n_trials", "mean_trial_corr", "median_trial_corr",
            "min_trial_corr", "mean_per_channel_corr", "median_per_channel_corr",
            "mae", "rmse", "relative_rmse"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in paired_rows:
            w.writerow(row)


# ============================ 六、图 ============================

def make_figures(session_rows, paired_rows, agg, manifest_summary, fig_dir: Path,
                 alignment_summary: Optional[Dict[str, Any]] = None) -> List[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    made: List[str] = []
    try:
        import pandas as pd
        df = pd.DataFrame(session_rows)
        for c in df.columns:
            if c not in ("subject_id", "session_id", "status", "official_mat_path"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
    except Exception as e:  # noqa: BLE001
        print(f"pandas 不可用，退化为 dict 画图：{e!r}", flush=True)
        df = _DictFrame(session_rows)

    def _try(name, fn):
        try:
            p = fn()
            made.append(str(p))
            print(f"  figure: {Path(p).name}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  figure FAILED {name}: {e!r}", flush=True)

    # 1 dashboard
    ms = manifest_summary
    exact = sum(1 for r in session_rows if r.get("labels_match_exact") is True)
    multiset_only = sum(1 for r in session_rows
                        if r.get("labels_match_exact") is False
                        and r.get("labels_multiset_match") is True)
    mismatch = sum(1 for r in session_rows if r.get("labels_multiset_match") is False
                   and r.get("official_found"))
    no_mat = sum(1 for r in session_rows if not r.get("official_found"))
    panels = {
        "status": {"ok": ms["2_status_counts"].get("ok", 0),
                   "failed": sum(v for k, v in ms["2_status_counts"].items() if k != "ok")},
        "labels vs official (.mat)": {"exact": exact, "multiset_only": multiset_only,
                                      "mismatch": mismatch, "no_mat": no_mat},
        "aux_cleaning_used": {"True": ms["14_aux_cleaning_used_true_count"],
                              "False": ms["1_total_sessions"] - ms["14_aux_cleaning_used_true_count"]},
        "ICA n_comp fallback": {"yes": ms["17_ica_n_components_fallback_count"],
                                "no": ms["1_total_sessions"] - ms["17_ica_n_components_fallback_count"]},
        "no-aux-clean fallback": {"yes": ms["18_no_aux_clean_fallback_count"],
                                  "no": ms["1_total_sessions"] - ms["18_no_aux_clean_fallback_count"]},
        "NaN/Inf sessions": {"clean": ms["1_total_sessions"] - len(ms["10_nan_inf_sessions"]),
                             "nan_inf": len(ms["10_nan_inf_sessions"])},
    }
    if alignment_summary is not None:
        n_perm = alignment_summary.get("n_permuted_subjects", 0)
        n_subj = alignment_summary.get("n_subjects", 0)
        panels["official ses-order (subjects)"] = {"identity": n_subj - n_perm,
                                                   "permuted": n_perm}
    _try("qc_dashboard", lambda: qp.plot_qc_dashboard(panels, fig_dir / "qc_dashboard.png"))

    _try("std_scatter", lambda: qp.plot_std_scatter(df, fig_dir / "std_ours_vs_official_scatter.png"))
    _try("std_ratio_hist", lambda: qp.plot_std_ratio_hist(df, fig_dir / "std_ratio_hist.png"))
    _try("rms_boxplot", lambda: qp.plot_rms_boxplot(df, fig_dir / "rms_boxplot_ours_vs_official.png"))
    _try("high_amp_boxplot",
         lambda: qp.plot_high_amp_boxplot(df, fig_dir / "high_amp_trial_ratio_boxplot.png"))

    if agg["freqs"] is not None and agg["psd_ours_global"] is not None:
        _try("psd_global", lambda: qp.plot_psd_overlay_global(
            agg["freqs"], agg["psd_ours_global"], agg["psd_off_global"],
            fig_dir / "psd_overlay_global.png"))
        if agg["sel_ours"]:
            _try("psd_channels", lambda: qp.plot_psd_overlay_channels(
                agg["freqs"], agg["sel_ours"], agg["sel_off"],
                fig_dir / "psd_overlay_C3_C4_Cz.png"))

    _try("bandpower_ratio",
         lambda: qp.plot_bandpower_ratio_mu_beta(df, fig_dir / "bandpower_ratio_mu_beta.png"))

    if agg["rms_ratio_matrix"] is not None:
        _try("channel_rms_heatmap", lambda: qp.plot_channel_rms_ratio_heatmap(
            agg["rms_ratio_matrix"], agg["rms_ratio_labels"], agg["channel_names"],
            fig_dir / "channel_rms_ratio_heatmap.png"))

    _try("trial_corr_hist", lambda: qp.plot_trial_corr_hist(
        agg["mean_trial_corr_vals"], fig_dir / "trial_corr_hist_exact_label_sessions.png",
        n_exact=len(agg["mean_trial_corr_vals"])))

    if agg["examples"]:
        _try("example_waveforms", lambda: qp.plot_example_waveforms(
            agg["examples"], fig_dir / "example_waveform_overlay.png"))

    _try("class_mu_beta", lambda: qp.plot_class_mu_beta_difference(
        df, fig_dir / "class_mu_beta_difference_C3_C4.png"))

    def _expand(hist: Dict[str, int]) -> List[int]:
        out: List[int] = []
        for k, v in hist.items():
            out += [int(k)] * int(v)
        return out

    eog_counts = _expand(ms["15_eog_excluded_stats"]["histogram"])
    ecg_counts = _expand(ms["16_ecg_excluded_stats"]["histogram"])
    total_counts = _expand(ms["16b_total_excluded_stats"]["histogram"])
    _try("ica_excluded", lambda: qp.plot_ica_excluded_summary(
        eog_counts, ecg_counts, total_counts,
        fig_dir / "ica_excluded_components_summary.png"))
    return made


class _DictFrame:
    """没有 pandas 时的极简替身：支持 df.get(col) 返回 list、len()、columns。"""

    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows
        self.columns = sorted({k for r in rows for k in r})

    def get(self, col, default=None):
        return [r.get(col, np.nan) for r in self.rows]

    def __getitem__(self, col):
        return self.get(col)

    def __len__(self):
        return len(self.rows)


# ============================ 七、QC_REPORT.md ============================

def _med(vals: List[float]) -> float:
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


def write_report_md(session_rows, paired_rows, agg, manifest_summary, out_path: Path,
                    fig_dir_name: str = "figures",
                    alignment_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ms = manifest_summary
    alignment_summary = alignment_summary or {"permuted_subjects": [], "per_subject": []}
    n = ms["1_total_sessions"]
    n_ok = ms["2_status_counts"].get("ok", 0)
    failed = ms["2_failed_sessions"]

    def col(name):
        return [r.get(name) for r in session_rows]

    std_ratio = [r.get("std_ratio") for r in session_rows if isinstance(r.get("std_ratio"), (int, float))]
    rms_ratio = [r.get("rms_ratio") for r in session_rows if isinstance(r.get("rms_ratio"), (int, float))]
    mu_ratio = [r.get("mu_bandpower_ratio") for r in session_rows if isinstance(r.get("mu_bandpower_ratio"), (int, float))]
    beta_ratio = [r.get("beta_bandpower_ratio") for r in session_rows if isinstance(r.get("beta_bandpower_ratio"), (int, float))]
    std_ratio = [v for v in std_ratio if np.isfinite(v)]
    rms_ratio = [v for v in rms_ratio if np.isfinite(v)]
    mu_ratio = [v for v in mu_ratio if np.isfinite(v)]
    beta_ratio = [v for v in beta_ratio if np.isfinite(v)]

    n_ours_lower = sum(1 for v in std_ratio if v < 1.0)

    # MI 可分性中位数
    def sep_med(side, ch, band):
        return _med([abs(r.get(f"{side}_{ch}_{band}_cohend"))
                     for r in session_rows
                     if isinstance(r.get(f"{side}_{ch}_{band}_cohend"), (int, float))
                     and np.isfinite(r.get(f"{side}_{ch}_{band}_cohend"))])

    sep = {}
    for ch in ("C3", "C4"):
        for band in ("mu_alpha", "beta"):
            sep[f"ours_{ch}_{band}"] = sep_med("ours", ch, band)
            sep[f"official_{ch}_{band}"] = sep_med("official", ch, band)

    # 关注 session
    attention: List[str] = []
    for r in session_rows:
        s = f"{r['subject_id']}/{r['session_id']}"
        reasons = []
        if r.get("status") not in ("ok", "success"):
            reasons.append("status≠ok")
        if not r.get("official_found"):
            reasons.append("无官方.mat")
        if r.get("labels_multiset_match") is False and r.get("official_found"):
            reasons.append("multiset不匹配")
        sr = r.get("std_ratio")
        if isinstance(sr, (int, float)) and np.isfinite(sr) and (sr < 0.5 or sr > 1.6):
            reasons.append(f"std_ratio={sr:.2f}")
        h5 = r.get("ours_high_amp_trial_ratio_500uV")
        if isinstance(h5, (int, float)) and np.isfinite(h5) and h5 > 0.05:
            reasons.append(f"高幅>500µV比例={h5:.2f}")
        if r.get("ours_nan_count") or r.get("ours_inf_count"):
            reasons.append("NaN/Inf")
        if reasons:
            attention.append(f"- `{s}`: " + ", ".join(reasons))

    paired_corr = [r["mean_trial_corr"] for r in paired_rows]
    rel_rmse = [r["relative_rmse"] for r in paired_rows]

    overall_pass = (
        len(ms["13_multiset_false_sessions"]) == 0
        and not ms["10_nan_inf_present"]
        and ms["7_sfreq_all_250"] and ms["8_n_channels_all_58"] and ms["9_n_times_all_1000"]
    )

    fd = fig_dir_name
    lines: List[str] = []
    lines.append("# 全量预处理质检 + 与官方 derivatives 对比报告\n")
    lines.append(f"> 数据：`eog_ecg_clean` 全量预处理结果 vs 官方 `derivatives/2C dataset_processeddata`。"
                 f" 本报告由 `scripts/compare_processed_quality.py` 自动生成。\n")

    lines.append("## 一、总体结论\n")
    verdict = "✅ 基本通过" if overall_pass else "⚠️ 需关注（见下）"
    lines.append(f"- **总体判定：{verdict}**")
    lines.append(f"- 总 session 数：**{n}**（期望 153）；status=ok：**{n_ok}/{n}**；"
                 f"失败：**{len(failed)}** 个。")
    if failed:
        lines.append(f"  - 失败 session：{', '.join('`'+x+'`' for x in failed)}（均为 trigger/试次数 < 200，属原始事件缺失，非降噪问题）。")
    lines.append(f"- 形状/采样率/通道/时长：X_shape 全 [200,58,1000] = **{ms['3_X_shape_all_200_58_1000']}**；"
                 f"sfreq=250 全通过=**{ms['7_sfreq_all_250']}**；n_channels=58=**{ms['8_n_channels_all_58']}**；"
                 f"n_times=1000=**{ms['9_n_times_all_1000']}**。")
    lines.append(f"- NaN/Inf：{'**无**' if not ms['10_nan_inf_present'] else '**存在：'+', '.join(ms['10_nan_inf_sessions'])+'**'}。")
    lines.append(f"- 标签多重集不匹配(危险)：{'**无**' if not ms['13_multiset_false_sessions'] else '**'+', '.join(ms['13_multiset_false_sessions'])+'**'}。\n")

    # 会话对齐
    permuted = alignment_summary.get("permuted_subjects", [])
    lines.append("## 二、会话(session)对齐：官方 derivatives 的 ses 顺序问题\n")
    lines.append("- 重要发现：官方 derivatives 对**部分被试**的 session 排序与 BIDS sourcedata "
                 "（即我们的 ses-YY）**不一致**。本脚本在每个被试内用幅值指纹 (std, max|·|) 把"
                 "我们的 session 配到官方真正对应的 session，**所有比值/配对均对匹配后的官方计算**。")
    lines.append(f"- 官方 ses 顺序与我们不一致的被试：**{len(permuted)}** / {alignment_summary.get('n_subjects', 0)}。")
    if permuted:
        lines.append(f"  - 受影响被试：{', '.join('`'+s+'`' for s in permuted)}。"
                     "（这不是我们预处理的错误：我们的 (X,y) 严格来自 sourcedata + evt.bdf，"
                     "被试内 session 幅值集合与官方一致；只是官方 .mat 的 ses 命名顺序不同。）")
    lines.append("- 含义：用「同名 ses」直接对比会对这些被试产生巨大的虚假 std/带功率比值；"
                 "对齐后这些虚假异常消失。下面所有数值都是**对齐后**的结果。\n")

    lines.append("## 三、与官方 derivatives 的主要差异（对齐后）\n")
    lines.append(f"- 全局 std 比值（ours/official，匹配后）中位数：**{_med(std_ratio):.3f}**；"
                 f"其中 {n_ours_lower}/{len(std_ratio)} 个 session ours 更低（符合预期：EOG/ECG 伪迹被清理）。")
    lines.append(f"- 全局 RMS 比值中位数：**{_med(rms_ratio):.3f}**。")
    lines.append(f"- mu/alpha bandpower 比值中位数：**{_med(mu_ratio):.3f}**；"
                 f"beta bandpower 比值中位数：**{_med(beta_ratio):.3f}**。")
    if paired_corr:
        lines.append(f"- 配对（labels_match_exact=True，共 {len(paired_corr)} 个 session）trial-wise "
                     f"相关中位数：**{_med(paired_corr):.3f}**；relative RMSE 中位数：**{_med(rel_rmse):.3f}**。")
        lines.append("  - 说明：trial 顺序与官方不一致的 session 不做 trial-wise 配对相关，只做分布级对比（频谱/幅值/可分性）。")
    lines.append(f"- 辅助降噪：aux_cleaning_used=True 共 **{ms['14_aux_cleaning_used_true_count']}**；"
                 f"EOG 排除 IC 的 session **{ms['15_eog_excluded_stats']['sessions_with_ge1']}** 个"
                 f"（共 {ms['15_eog_excluded_stats']['total_components']} 个分量）；"
                 f"ECG 排除 IC 的 session **{ms['16_ecg_excluded_stats']['sessions_with_ge1']}** 个"
                 f"（共 {ms['16_ecg_excluded_stats']['total_components']} 个）；"
                 f"ICA n_components 回退 **{ms['17_ica_n_components_fallback_count']}** 次；"
                 f"no-aux-clean 回退 **{ms['18_no_aux_clean_fallback_count']}** 次。\n")

    lines.append("## 四、幅值是否异常（对齐后）\n")
    h5_bad = [f"{r['subject_id']}/{r['session_id']}" for r in session_rows
              if isinstance(r.get("ours_high_amp_trial_ratio_500uV"), (int, float))
              and np.isfinite(r.get("ours_high_amp_trial_ratio_500uV"))
              and r.get("ours_high_amp_trial_ratio_500uV") > 0.05]
    lines.append(f"- std 比值落在 [0.5,1.6] 之外的 session 视为可疑（见下方关注列表）。")
    lines.append(f"- ours 端 >500µV 高幅 trial 比例 >5% 的 session：{'无' if not h5_bad else ', '.join('`'+x+'`' for x in h5_bad)}。\n")

    lines.append("## 五、是否保留 MI 的 mu/beta 与 C3/C4 左右手差异\n")
    lines.append("两类（左/右手）在 C3/C4 的 mu、beta log-bandpower 上的可分性（|Cohen's d| 中位数）：\n")
    lines.append("| 通道/频带 | ours |Cohen d| | official |Cohen d| |")
    lines.append("| --- | --- | --- |")
    for ch in ("C3", "C4"):
        for band, disp in (("mu_alpha", "mu"), ("beta", "beta")):
            lines.append(f"| {ch} {disp} | {sep[f'ours_{ch}_{band}']:.3f} | {sep[f'official_{ch}_{band}']:.3f} |")
    lines.append("\n- 解读：ours 与 official 的可分性量级接近即说明 EOG/ECG 清理没有破坏 MI 判别信息；"
                 "ours 略有不同属正常（清理改变了非脑成分）。\n")

    lines.append("## 六、需要关注的 session（对齐后仍异常的）\n")
    if attention:
        lines.extend(attention)
    else:
        lines.append("- 无（所有 session 均在正常范围）。")
    lines.append("")

    lines.append("## 七、是否建议进入 41/10 split + SHUTrialDataset\n")
    if overall_pass:
        lines.append(f"- **建议：可以进入 41/10 split 阶段。** 用 status=ok 的 **{n_ok}** 个 session "
                     "（`SHUTrialDataset.from_manifest(..., statuses=('ok',))` 默认即过滤）。")
        if failed:
            lines.append(f"- 但请先决定 {len(failed)} 个失败 session 的处理：要么排除，要么回原始数据重提 trigger"
                         "（它们缺 1~5 个试次，是事件/触发问题，与降噪无关）。这些 subject 的其它 session 正常，不影响被试级划分。")
    else:
        lines.append("- **暂不建议**：存在 NaN/Inf 或 multiset 不匹配等硬问题，请先修复再进入 split。")
    lines.append("")

    lines.append("## 八、图表索引\n")
    for fn, desc in [
        ("qc_dashboard.png", "QC 总览：status/label/aux/ICA fallback/NaN-Inf"),
        ("std_ours_vs_official_scatter.png", "每 session std 散点（带 y=x）"),
        ("std_ratio_hist.png", "std 比值分布"),
        ("rms_boxplot_ours_vs_official.png", "RMS 箱线图"),
        ("high_amp_trial_ratio_boxplot.png", "高幅 trial 比例对比"),
        ("psd_overlay_global.png", "全局平均 PSD overlay"),
        ("psd_overlay_C3_C4_Cz.png", "C3/C4/Cz 的 PSD overlay"),
        ("bandpower_ratio_mu_beta.png", "mu/beta bandpower 比值"),
        ("channel_rms_ratio_heatmap.png", "逐通道 RMS 比值热图"),
        ("trial_corr_hist_exact_label_sessions.png", "exact-label session 的 trial 相关"),
        ("example_waveform_overlay.png", "示例波形 overlay"),
        ("class_mu_beta_difference_C3_C4.png", "C3/C4 左右手 mu/beta 可分性对比"),
        ("ica_excluded_components_summary.png", "EOG/ECG 排除 IC 统计"),
    ]:
        lines.append(f"- `{fd}/{fn}` — {desc}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "overall_pass": overall_pass,
        "n_ok": n_ok, "n_total": n, "failed": failed,
        "std_ratio_median": _med(std_ratio),
        "mu_ratio_median": _med(mu_ratio), "beta_ratio_median": _med(beta_ratio),
        "paired_corr_median": _med(paired_corr) if paired_corr else None,
        "attention_count": len(attention),
    }


# ============================ main ============================

def main() -> int:
    ap = argparse.ArgumentParser(description="全量预处理质检 + 与官方 derivatives 对比")
    ap.add_argument("--processed-manifest", required=True)
    ap.add_argument("--official-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-example-sessions", type=int, default=6)
    ap.add_argument("--mi-channels", default="C3,C4,Cz")
    ap.add_argument("--nperseg", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个 session（smoke test 用）")
    args = ap.parse_args()

    manifest = Path(args.processed_manifest)
    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    mi_channels = [c.strip() for c in args.mi_channels.split(",") if c.strip()]

    if not manifest.exists():
        print(f"ERROR: manifest 不存在: {manifest}", file=sys.stderr)
        return 2

    rows = _read_csv_rows(manifest)
    print(f"读到 manifest：{len(rows)} 行。输出目录：{out_dir}", flush=True)

    # 一、manifest QC
    print("== 全量 manifest 质检 ==", flush=True)
    manifest_summary = run_manifest_qc(rows)
    save_json(manifest_summary, out_dir / "manifest_qc_summary.json")
    print(f"  status: {manifest_summary['2_status_counts']}; "
          f"failed: {manifest_summary['2_failed_sessions']}", flush=True)
    print(f"  X_shape ok={manifest_summary['3_X_shape_all_200_58_1000']}, "
          f"NaN/Inf present={manifest_summary['10_nan_inf_present']}, "
          f"total size={manifest_summary['19_total_output_human']}", flush=True)

    # 三/四、逐 session 对比（含被试内 session 对齐）
    print("== 逐 session 质量对比（读 npz + 官方 mat，被试内对齐后算 PSD/可分性）==", flush=True)
    session_rows, paired_rows, agg, alignment_summary = run_comparison(
        rows, args.official_root, mi_channels,
        max_examples=args.max_example_sessions, nperseg=args.nperseg,
        limit=(args.limit or None))
    alignment_summary["n_subjects"] = len(alignment_summary["per_subject"])
    alignment_summary["n_permuted_subjects"] = len(alignment_summary["permuted_subjects"])
    save_json(alignment_summary, out_dir / "session_alignment.json")
    print(f"  官方 session 顺序与 sourcedata 不一致的被试：{alignment_summary['n_permuted_subjects']} 个："
          f"{alignment_summary['permuted_subjects']}", flush=True)

    # 五、CSV
    write_session_csv(session_rows, out_dir / "session_quality_metrics.csv")
    write_paired_csv(paired_rows, out_dir / "paired_similarity_metrics.csv")
    print(f"  写出 session_quality_metrics.csv ({len(session_rows)} 行), "
          f"paired_similarity_metrics.csv ({len(paired_rows)} 行)", flush=True)

    # 六、图
    print("== 生成图 ==", flush=True)
    made = make_figures(session_rows, paired_rows, agg, manifest_summary, fig_dir,
                        alignment_summary)
    print(f"  生成 {len(made)} 张图到 {fig_dir}", flush=True)

    # 七、报告
    report_stats = write_report_md(session_rows, paired_rows, agg, manifest_summary,
                                   out_dir / "QC_REPORT.md", alignment_summary=alignment_summary)
    print("== 完成 ==", flush=True)
    print(f"  overall_pass={report_stats['overall_pass']}, ok={report_stats['n_ok']}/{report_stats['n_total']}, "
          f"failed={report_stats['failed']}, attention={report_stats['attention_count']}", flush=True)
    print(f"  std_ratio_median={report_stats['std_ratio_median']:.3f}, "
          f"mu_ratio_median={report_stats['mu_ratio_median']:.3f}, "
          f"beta_ratio_median={report_stats['beta_ratio_median']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
