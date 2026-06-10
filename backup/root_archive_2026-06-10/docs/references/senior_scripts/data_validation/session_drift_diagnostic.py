#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Session 漂移诊断脚本 (Session Drift Diagnostic)
================================================
用途：量化跨 session EEG 分布漂移，回答"跨 session 泛化为什么困难"
输入：预处理后的 .npz 文件目录（每个文件含 X[trials, channels, timepoints], y[trials]）
输出：漂移指标 CSV 报告 + 可视化 PNG

使用方式：
    python session_drift_diagnostic.py --data_dir /path/to/eog_ecg_clean/ --output_dir ./drift_report

依赖：
    numpy >= 1.21
    scipy >= 1.7
    scikit-learn >= 1.0
    mne >= 1.0
    matplotlib >= 3.5
    seaborn >= 0.12
"""

import os
import re
import glob
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ============================================================
# 常量定义
# ============================================================
FS = 250  # 采样率 Hz
MU_BAND = (8, 13)   # μ 频段 Hz
BETA_BAND = (13, 30) # β 频段 Hz
FAILED_SESSIONS = {
    ("sub-023", "ses-01"), ("sub-024", "ses-02"), ("sub-024", "ses-03"),
    ("sub-026", "ses-01"), ("sub-032", "ses-02"),
}


# ============================================================
# 文件解析
# ============================================================
def parse_filename(filepath):
    """从文件名解析 subject_id 和 session_id。

    支持格式：sub-XXX_ses-YY.npz, subXXX_sesYY.npz, S01_Session1.npz 等
    返回 (subject_id, session_id) 或 None
    """
    basename = Path(filepath).stem
    # 格式1: sub-XXX_ses-YY
    m = re.search(r"(sub-\d+)[_\-](ses-\d+)", basename)
    if m:
        return m.group(1), m.group(2)
    # 格式2: subXXX_sesYY
    m = re.search(r"(sub\d+)[_\-](ses\d+)", basename, re.IGNORECASE)
    if m:
        return m.group(1), m.group(2)
    # 格式3: SXX_SessionY
    m = re.search(r"S(\d+)[_\-]Session(\d+)", basename, re.IGNORECASE)
    if m:
        return f"sub-{m.group(1)}", f"ses-{m.group(2)}"
    return None


def load_sessions(data_dir):
    """扫描目录，加载所有 .npz 文件，按 subject/session 分组。

    返回 dict: {(subject_id, session_id): {"X": ndarray, "y": ndarray}}
    """
    npz_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.npz"), recursive=True))
    if not npz_files:
        raise FileNotFoundError(f"未找到 .npz 文件: {data_dir}")

    sessions = {}
    skipped = 0
    for fp in npz_files:
        parsed = parse_filename(fp)
        if parsed is None:
            print(f"  [跳过] 无法解析文件名: {Path(fp).name}")
            skipped += 1
            continue
        sub_id, ses_id = parsed
        if (sub_id, ses_id) in FAILED_SESSIONS:
            print(f"  [跳过] failed session: {sub_id}/{ses_id}")
            skipped += 1
            continue

        data = np.load(fp)
        X = data["X"]  # [trials, channels, timepoints]
        y = data["y"]  # [trials]
        sessions[(sub_id, ses_id)] = {"X": X, "y": y}

    print(f"加载完成: {len(sessions)} 个 session, 跳过 {skipped} 个")
    return sessions


# ============================================================
# 频域特征提取
# ============================================================
def compute_band_power(X, fs=FS, band=MU_BAND):
    """计算每个 trial 每个通道的频带功率。

    参数:
        X: ndarray [trials, channels, timepoints]
        band: (low, high) Hz
    返回:
        power: ndarray [trials, channels] — 对数功率
    """
    n_trials, n_chans, n_times = X.shape
    power = np.zeros((n_trials, n_chans))
    freqs = np.fft.rfftfreq(n_times, d=1.0 / fs)
    band_mask = (freqs >= band[0]) & (freqs <= band[1])

    for i in range(n_trials):
        for j in range(n_chans):
            fft_vals = np.fft.rfft(X[i, j, :])
            psd = np.abs(fft_vals) ** 2
            power[i, j] = np.log10(np.mean(psd[band_mask]) + 1e-10)
    return power


def compute_erd_ers(X, fs=FS, band=MU_BAND, baseline_ratio=0.25):
    """计算 ERD/ERS（事件相关去同步/同步化）。

    以 trial 前 baseline_ratio 部分为基线，计算 MI 期间的相对功率变化。
    返回:
        erd: ndarray [channels] — 平均 ERD/ERS 百分比（负=去同步，正=同步化）
    """
    n_trials, n_chans, n_times = X.shape
    bl_end = int(n_times * baseline_ratio)
    freqs = np.fft.rfftfreq(n_times, d=1.0 / fs)
    band_mask = (freqs >= band[0]) & (freqs <= band[1])

    erd_all = np.zeros((n_trials, n_chans))
    for i in range(n_trials):
        for j in range(n_chans):
            # 基线功率
            bl_fft = np.fft.rfft(X[i, j, :bl_end])
            bl_psd = np.abs(bl_fft) ** 2
            bl_power = np.mean(bl_psd[band_mask[:len(bl_fft)]] if len(bl_fft) < len(band_mask) else bl_psd[band_mask]) + 1e-10
            # MI 功率
            mi_fft = np.fft.rfft(X[i, j, bl_end:])
            mi_psd = np.abs(mi_fft) ** 2
            mi_band = band_mask[:len(mi_fft)] if len(mi_fft) < len(band_mask) else band_mask
            mi_power = np.mean(mi_psd[mi_band]) + 1e-10
            # ERD/ERS 百分比
            erd_all[i, j] = (mi_power - bl_power) / bl_power * 100

    return np.median(erd_all, axis=0)  # [channels]，取中位数更稳健


# ============================================================
# 分布距离指标
# ============================================================
def compute_mmd(X1, X2, kernel="rbf", gamma=None):
    """计算两组样本的 MMD (Maximum Mean Discrepancy)。

    将 trial × (channels × timepoints) 展平后计算。
    """
    # 展平为 2D: [trials, features]
    flat1 = X1.reshape(X1.shape[0], -1)
    flat2 = X2.reshape(X2.shape[0], -1)

    # 子采样以加速（每组最多 100 个 trial）
    n_sub = min(100, flat1.shape[0], flat2.shape[0])
    idx1 = np.random.choice(flat1.shape[0], n_sub, replace=False)
    idx2 = np.random.choice(flat2.shape[0], n_sub, replace=False)
    flat1 = flat1[idx1]
    flat2 = flat2[idx2]

    if gamma is None:
        # 中位数启发式
        D = cdist(flat1, flat2, "sqeuclidean")
        gamma = 1.0 / np.median(D[D > 0])

    K11 = np.exp(-gamma * cdist(flat1, flat1, "sqeuclidean"))
    K22 = np.exp(-gamma * cdist(flat2, flat2, "sqeuclidean"))
    K12 = np.exp(-gamma * cdist(flat1, flat2, "sqeuclidean"))

    mmd_sq = np.mean(K11) + np.mean(K22) - 2 * np.mean(K12)
    return np.sqrt(max(mmd_sq, 0))


def compute_coral(X1, X2):
    """计算 CORAL 距离（协方差矩阵 Frobenius 范数差）。

    使用通道级统计量（均值 + 协方差），避免展平后维度爆炸。
    """
    # 对每个 trial 计算通道级统计
    def trial_stats(X):
        # X: [trials, channels, timepoints]
        mean_per_trial = X.mean(axis=2)  # [trials, channels]
        return mean_per_trial

    stats1 = trial_stats(X1)  # [trials, channels]
    stats2 = trial_stats(X2)

    C1 = np.cov(stats1.T)  # [channels, channels]
    C2 = np.cov(stats2.T)

    # 正则化避免奇异
    eps = 1e-6
    C1 += eps * np.eye(C1.shape[0])
    C2 += eps * np.eye(C2.shape[0])

    diff = C1 - C2
    return np.sqrt(np.sum(diff ** 2)) / diff.shape[0]


def compute_fisher_ratio(X, y):
    """计算 Fisher 判别比（类间方差 / 类内方差）。

    使用通道级 μ 功率作为特征。
    """
    mu_power = compute_band_power(X, band=MU_BAND)  # [trials, channels]
    classes = np.unique(y)
    if len(classes) < 2:
        return 0.0

    # 对每个通道计算 Fisher ratio，取平均
    fisher_per_chan = []
    for ch in range(mu_power.shape[1]):
        feat = mu_power[:, ch]
        class_means = [feat[y == c].mean() for c in classes]
        class_vars = [feat[y == c].var() for c in classes]
        class_counts = [np.sum(y == c) for c in classes]

        overall_mean = feat.mean()
        between = sum(n * (m - overall_mean) ** 2 for n, m in zip(class_counts, class_means))
        within = sum(n * v for n, v in zip(class_counts, class_vars))

        fisher_per_chan.append(between / (within + 1e-10))

    return np.mean(fisher_per_chan)


# ============================================================
# 空间模式指标
# ============================================================
def compute_csp_similarity(X1, y1, X2, y2, n_components=4):
    """计算两个 session 的 CSP 模式余弦相似度。

    在每个 session 上独立拟合 CSP，比较空间滤波器的方向一致性。
    """
    try:
        from sklearn.base import BaseEstimator, TransformerMixin

        class SimpleCSP(BaseEstimator, TransformerMixin):
            """简化的 CSP 实现（避免依赖 mne.decoding）。"""
            def __init__(self, n_components=4):
                self.n_components = n_components
                self.filters_ = None

            def fit(self, X, y):
                # X: [trials, channels, timepoints]
                classes = np.unique(y)
                if len(classes) != 2:
                    raise ValueError("CSP 需要二分类")

                # 计算每类的平均协方差
                covs = []
                for c in classes:
                    X_c = X[y == c]  # [n_c, channels, timepoints]
                    # 展平时间维度后计算协方差
                    flat = X_c.transpose(1, 0, 2).reshape(X_c.shape[1], -1)  # [channels, n_c*times]
                    cov = np.cov(flat)
                    cov += 1e-6 * np.eye(cov.shape[0])
                    covs.append(cov)

                # 广义特征值分解
                C_sum = covs[0] + covs[1]
                eigvals, eigvecs = np.linalg.eigh(np.linalg.inv(C_sum) @ covs[0])
                # 按特征值排序，取最大和最小的 n_components/2 个
                idx = np.argsort(eigvals)[::-1]
                n_half = self.n_components // 2
                self.filters_ = eigvecs[:, np.concatenate([idx[:n_half], idx[-n_half:]])]
                return self

            def transform(self, X):
                # X: [trials, channels, timepoints]
                projected = np.einsum("cf,tcn->tfn", self.filters_.T, X)
                # 计算每个 trial 的 log 方差作为特征
                features = np.log(np.var(projected, axis=2) + 1e-10)
                return features

        csp1 = SimpleCSP(n_components=n_components).fit(X1, y1)
        csp2 = SimpleCSP(n_components=n_components).fit(X2, y2)

        # 计算 CSP 滤波器的余弦相似度
        f1 = csp1.filters_  # [channels, n_components]
        f2 = csp2.filters_

        # 对每对分量计算 |cosine similarity|，取最大匹配
        similarities = []
        for i in range(f1.shape[1]):
            for j in range(f2.shape[1]):
                cos_sim = np.abs(np.dot(f1[:, i], f2[:, j]) /
                                 (np.linalg.norm(f1[:, i]) * np.linalg.norm(f2[:, j]) + 1e-10))
                similarities.append(cos_sim)

        # 取 top n_components 个最大相似度的平均
        similarities = sorted(similarities, reverse=True)
        return np.mean(similarities[:n_components])

    except Exception as e:
        print(f"  [警告] CSP 计算失败: {e}")
        return np.nan


def compute_channel_rms_ratio(X1, X2):
    """计算两个 session 的通道级 RMS 比值。

    返回:
        ratio: ndarray [channels] — RMS(session2) / RMS(session1)
    """
    rms1 = np.sqrt(np.mean(X1 ** 2, axis=(0, 2)))  # [channels]
    rms2 = np.sqrt(np.mean(X2 ** 2, axis=(0, 2)))
    return rms2 / (rms1 + 1e-10)


# ============================================================
# 信号质量指标
# ============================================================
def compute_signal_quality(X):
    """计算信号质量指标。

    返回 dict:
        high_amp_ratio: 高幅 trial 比例（幅值 > 100 μV）
        mean_rms: 平均 RMS
        std_rms: RMS 标准差
    """
    rms_per_trial = np.sqrt(np.mean(X ** 2, axis=(1, 2)))  # [trials]
    high_amp = np.max(np.abs(X), axis=(1, 2))
    return {
        "high_amp_ratio": np.mean(high_amp > 100),
        "mean_rms": np.mean(rms_per_trial),
        "std_rms": np.std(rms_per_trial),
    }


# ============================================================
# 主诊断流程
# ============================================================
def run_diagnostic(data_dir, output_dir):
    """执行完整的 session 漂移诊断。"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    # 1. 加载数据
    print("=" * 60)
    print("Step 1: 加载数据")
    print("=" * 60)
    sessions = load_sessions(data_dir)

    # 按 subject 分组
    subjects = defaultdict(dict)
    for (sub_id, ses_id), data in sessions.items():
        subjects[sub_id][ses_id] = data

    print(f"共 {len(subjects)} 个被试, {len(sessions)} 个 session")

    # 2. 计算指标
    print("\n" + "=" * 60)
    print("Step 2: 计算跨 session 漂移指标")
    print("=" * 60)

    results = []
    pair_count = 0

    for sub_id in sorted(subjects.keys()):
        ses_dict = subjects[sub_id]
        ses_ids = sorted(ses_dict.keys())

        if len(ses_ids) < 2:
            print(f"  {sub_id}: 仅 {len(ses_ids)} 个 session，跳过")
            continue

        # 计算每对 session 的指标
        for i in range(len(ses_ids)):
            for j in range(i + 1, len(ses_ids)):
                ses_i, ses_j = ses_ids[i], ses_ids[j]
                Xi = ses_dict[ses_i]["X"]
                yi = ses_dict[ses_i]["y"]
                Xj = ses_dict[ses_j]["X"]
                yj = ses_dict[ses_j]["y"]

                pair_count += 1
                if pair_count % 10 == 0:
                    print(f"  处理中: {pair_count} 对 ...")

                row = {"subject": sub_id, "ses_i": ses_i, "ses_j": ses_j}

                # 分布距离
                row["mmd"] = compute_mmd(Xi, Xj)
                row["coral"] = compute_coral(Xi, Xj)

                # 频域漂移
                mu_i = compute_band_power(Xi, band=MU_BAND)
                mu_j = compute_band_power(Xj, band=MU_BAND)
                beta_i = compute_band_power(Xi, band=BETA_BAND)
                beta_j = compute_band_power(Xj, band=BETA_BAND)
                row["mu_power_shift"] = np.mean(mu_j) - np.mean(mu_i)
                row["beta_power_shift"] = np.mean(beta_j) - np.mean(beta_i)

                # μ 功率分布的 KS 检验
                row["mu_ks_stat"] = stats.ks_2samp(
                    mu_i.flatten(), mu_j.flatten()
                ).statistic

                # ERD/ERS 模式
                erd_i = compute_erd_ers(Xi, band=MU_BAND)
                erd_j = compute_erd_ers(Xj, band=MU_BAND)
                row["erd_mu_corr"] = np.corrcoef(erd_i, erd_j)[0, 1]

                erd_beta_i = compute_erd_ers(Xi, band=BETA_BAND)
                erd_beta_j = compute_erd_ers(Xj, band=BETA_BAND)
                row["erd_beta_corr"] = np.corrcoef(erd_beta_i, erd_beta_j)[0, 1]

                # CSP 模式相似度
                row["csp_similarity"] = compute_csp_similarity(Xi, yi, Xj, yj)

                # 通道 RMS 比值
                rms_ratio = compute_channel_rms_ratio(Xi, Xj)
                row["rms_ratio_median"] = np.median(rms_ratio)
                row["rms_ratio_std"] = np.std(rms_ratio)

                # Fisher 判别比
                row["fisher_i"] = compute_fisher_ratio(Xi, yi)
                row["fisher_j"] = compute_fisher_ratio(Xj, yj)
                row["fisher_shift"] = row["fisher_j"] - row["fisher_i"]

                # 信号质量
                qi = compute_signal_quality(Xi)
                qj = compute_signal_quality(Xj)
                row["high_amp_ratio_i"] = qi["high_amp_ratio"]
                row["high_amp_ratio_j"] = qj["high_amp_ratio"]
                row["mean_rms_i"] = qi["mean_rms"]
                row["mean_rms_j"] = qj["mean_rms"]

                results.append(row)

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "session_drift_report.csv")
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\n报告已保存: {csv_path}")

    # 3. 生成可视化
    print("\n" + "=" * 60)
    print("Step 3: 生成可视化")
    print("=" * 60)
    generate_figures(df, output_dir)

    # 4. 打印摘要
    print("\n" + "=" * 60)
    print("Step 4: 诊断摘要")
    print("=" * 60)
    print_summary(df)

    return df


# ============================================================
# 可视化
# ============================================================
def generate_figures(df, output_dir):
    """生成漂移诊断可视化图表。"""
    fig_dir = os.path.join(output_dir, "figures")
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # --- 图1: 分布距离热力图 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, title in zip(axes, ["mmd", "coral"], ["MMD 距离", "CORAL 距离"]):
        pivot = df.pivot_table(index="subject", columns="ses_i", values=metric, aggfunc="mean")
        sns.heatmap(pivot, ax=ax, cmap="YlOrRd", annot=False)
        ax.set_title(f"跨 Session {title}")
        ax.set_xlabel("Session")
        ax.set_ylabel("Subject")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "distribution_distance_heatmap.png"), dpi=150)
    plt.close()
    print("  [图1] distribution_distance_heatmap.png")

    # --- 图2: 频域漂移分布 ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, label in zip(axes, ["mu_power_shift", "beta_power_shift"], ["μ 功率漂移", "β 功率漂移"]):
        ax.hist(df[metric].dropna(), bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="red", linestyle="--", label="零漂移")
        ax.set_xlabel(f"{label} (log₁₀)")
        ax.set_ylabel("Session pair 数量")
        ax.set_title(f"{label} 分布")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "band_power_shift_hist.png"), dpi=150)
    plt.close()
    print("  [图2] band_power_shift_hist.png")

    # --- 图3: ERD/ERS 模式相关性 ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["erd_mu_corr"].dropna(), bins=20, alpha=0.6, label="μ ERD/ERS", edgecolor="black")
    ax.hist(df["erd_beta_corr"].dropna(), bins=20, alpha=0.6, label="β ERD/ERS", edgecolor="black")
    ax.axvline(1.0, color="red", linestyle="--", label="完美一致")
    ax.set_xlabel("Session 间 ERD/ERS 空间模式相关系数")
    ax.set_ylabel("Session pair 数量")
    ax.set_title("跨 Session ERD/ERS 模式一致性")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "erd_ers_correlation_hist.png"), dpi=150)
    plt.close()
    print("  [图3] erd_ers_correlation_hist.png")

    # --- 图4: CSP 模式相似度 ---
    fig, ax = plt.subplots(figsize=(8, 5))
    valid = df["csp_similarity"].dropna()
    if len(valid) > 0:
        ax.hist(valid, bins=20, edgecolor="black", alpha=0.7)
        ax.axvline(1.0, color="red", linestyle="--", label="完美一致")
        ax.axvline(valid.mean(), color="blue", linestyle="-", label=f"均值={valid.mean():.3f}")
        ax.set_xlabel("CSP 模式余弦相似度")
        ax.set_ylabel("Session pair 数量")
        ax.set_title("跨 Session CSP 空间模式一致性")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "csp_similarity_hist.png"), dpi=150)
    plt.close()
    print("  [图4] csp_similarity_hist.png")

    # --- 图5: Fisher 判别比漂移 ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["fisher_i"], df["fisher_j"], alpha=0.5, s=20)
    lim = max(df["fisher_i"].max(), df["fisher_j"].max()) * 1.1
    ax.plot([0, lim], [0, lim], "r--", label="y=x（无变化）")
    ax.set_xlabel("Session i Fisher 判别比")
    ax.set_ylabel("Session j Fisher 判别比")
    ax.set_title("跨 Session MI 可分性变化")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fisher_ratio_scatter.png"), dpi=150)
    plt.close()
    print("  [图5] fisher_ratio_scatter.png")

    # --- 图6: 通道 RMS 比值热力图 ---
    # 取所有被试的平均 RMS 比值（需要逐通道，这里用 summary 统计）
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["rms_ratio_median"].dropna(), bins=20, edgecolor="black", alpha=0.7)
    ax.axvline(1.0, color="red", linestyle="--", label="无变化")
    ax.set_xlabel("通道 RMS 比值中位数 (Sj / Si)")
    ax.set_ylabel("Session pair 数量")
    ax.set_title("跨 Session 信号幅值一致性")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "rms_ratio_hist.png"), dpi=150)
    plt.close()
    print("  [图6] rms_ratio_hist.png")

    # --- 图7: 指标相关性矩阵 ---
    metrics = ["mmd", "coral", "mu_power_shift", "erd_mu_corr", "csp_similarity",
               "fisher_shift", "rms_ratio_median"]
    available = [m for m in metrics if m in df.columns and df[m].notna().sum() > 5]
    if len(available) >= 3:
        fig, ax = plt.subplots(figsize=(8, 6))
        corr = df[available].corr()
        sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                    vmin=-1, vmax=1, square=True)
        ax.set_title("漂移指标相关性矩阵")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "metric_correlation_matrix.png"), dpi=150)
        plt.close()
        print("  [图7] metric_correlation_matrix.png")

    # --- 图8: 学习效应（如果 session 有序） ---
    df_sorted = df.copy()
    df_sorted["ses_i_num"] = df_sorted["ses_i"].str.extract(r"(\d+)").astype(int)
    df_sorted["ses_j_num"] = df_sorted["ses_j"].str.extract(r"(\d+)").astype(int)
    df_sorted["pair_type"] = df_sorted["ses_i_num"].astype(str) + "-" + df_sorted["ses_j_num"].astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, label in zip(axes, ["mmd", "fisher_shift"], ["MMD 距离", "Fisher 判别比变化"]):
        sns.boxplot(data=df_sorted, x="pair_type", y=metric, ax=ax)
        ax.set_xlabel("Session 对")
        ax.set_ylabel(label)
        ax.set_title(f"不同 Session 对的 {label}")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "session_pair_comparison.png"), dpi=150)
    plt.close()
    print("  [图8] session_pair_comparison.png")

    print(f"\n所有图表已保存至: {fig_dir}")


# ============================================================
# 摘要输出
# ============================================================
def print_summary(df):
    """打印诊断摘要。"""
    print("\n【分布距离】")
    print(f"  MMD  — 均值: {df['mmd'].mean():.4f}, 中位数: {df['mmd'].median():.4f}, "
          f"标准差: {df['mmd'].std():.4f}")
    print(f"  CORAL — 均值: {df['coral'].mean():.4f}, 中位数: {df['coral'].median():.4f}")

    print("\n【频域漂移】")
    print(f"  μ 功率漂移 — 均值: {df['mu_power_shift'].mean():.4f} (log₁₀), "
          f"标准差: {df['mu_power_shift'].std():.4f}")
    print(f"  β 功率漂移 — 均值: {df['beta_power_shift'].mean():.4f}, "
          f"标准差: {df['beta_power_shift'].std():.4f}")
    print(f"  μ KS 统计量 — 均值: {df['mu_ks_stat'].mean():.4f}")

    print("\n【ERD/ERS 模式一致性】")
    print(f"  μ ERD/ERS 相关 — 均值: {df['erd_mu_corr'].mean():.3f}, "
          f"中位数: {df['erd_mu_corr'].median():.3f}")
    print(f"  β ERD/ERS 相关 — 均值: {df['erd_beta_corr'].mean():.3f}")

    csp_valid = df["csp_similarity"].dropna()
    if len(csp_valid) > 0:
        print("\n【CSP 空间模式】")
        print(f"  余弦相似度 — 均值: {csp_valid.mean():.3f}, "
              f"中位数: {csp_valid.median():.3f}")

    print("\n【MI 可分性 (Fisher 判别比)】")
    print(f"  Session i — 均值: {df['fisher_i'].mean():.3f}")
    print(f"  Session j — 均值: {df['fisher_j'].mean():.3f}")
    print(f"  漂移 (j-i) — 均值: {df['fisher_shift'].mean():.3f}, "
          f"标准差: {df['fisher_shift'].std():.3f}")

    print("\n【信号幅值一致性】")
    print(f"  RMS 比值中位数 — 均值: {df['rms_ratio_median'].mean():.3f}, "
          f"标准差: {df['rms_ratio_median'].std():.3f}")

    print("\n【信号质量】")
    print(f"  高幅 trial 比例 — Session i: {df['high_amp_ratio_i'].mean():.3f}, "
          f"Session j: {df['high_amp_ratio_j'].mean():.3f}")


# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Session 漂移诊断")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="预处理 .npz 文件目录")
    parser.add_argument("--output_dir", type=str, default="./drift_report",
                        help="输出目录（默认: ./drift_report）")
    args = parser.parse_args()

    print("=" * 60)
    print("Session 漂移诊断")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)

    df = run_diagnostic(args.data_dir, args.output_dir)
    print(f"\n共计算 {len(df)} 个 session pair 的漂移指标。")
