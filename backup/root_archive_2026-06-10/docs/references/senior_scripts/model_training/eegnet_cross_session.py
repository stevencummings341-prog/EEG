#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EEGNet Cross-Session Baseline
==============================
用途：建立 MI 跨 session/跨被试分类基线，量化"难到什么程度"
输入：预处理后的 .npz 文件目录
输出：准确率 CSV + 可视化

使用方式：
    # Within-session 10-fold CV
    python eegnet_cross_session.py --data_dir /path/to/data --protocol within

    # Cross-session (train on ses-i, test on ses-j)
    python eegnet_cross_session.py --data_dir /path/to/data --protocol cross

    # LOSO (leave-one-subject-out)
    python eegnet_cross_session.py --data_dir /path/to/data --protocol loso

    # 全部运行
    python eegnet_cross_session.py --data_dir /path/to/data --protocol all

依赖：
    numpy >= 1.21
    torch >= 1.12
    scikit-learn >= 1.0
    pandas >= 1.4
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ============================================================
# 常量
# ============================================================
FAILED_SESSIONS = {
    ("sub-023", "ses-01"), ("sub-024", "ses-02"), ("sub-024", "ses-03"),
    ("sub-026", "ses-01"), ("sub-032", "ses-02"),
}


# ============================================================
# 文件解析（复用 session_drift_diagnostic 的逻辑）
# ============================================================
def parse_filename(filepath):
    """从文件名解析 subject_id 和 session_id。"""
    basename = Path(filepath).stem
    m = re.search(r"(sub-\d+)[_\-](ses-\d+)", basename)
    if m:
        return m.group(1), m.group(2)
    m = re.search(r"(sub\d+)[_\-](ses\d+)", basename, re.IGNORECASE)
    if m:
        return m.group(1), m.group(2)
    m = re.search(r"S(\d+)[_\-]Session(\d+)", basename, re.IGNORECASE)
    if m:
        return f"sub-{m.group(1)}", f"ses-{m.group(2)}"
    return None


def load_sessions(data_dir):
    """加载所有 .npz 文件，按 subject/session 分组。"""
    npz_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.npz"), recursive=True))
    if not npz_files:
        raise FileNotFoundError(f"未找到 .npz 文件: {data_dir}")

    sessions = {}
    for fp in npz_files:
        parsed = parse_filename(fp)
        if parsed is None:
            continue
        sub_id, ses_id = parsed
        if (sub_id, ses_id) in FAILED_SESSIONS:
            continue
        data = np.load(fp)
        sessions[(sub_id, ses_id)] = {"X": data["X"], "y": data["y"]}

    print(f"加载完成: {len(sessions)} 个 session")
    return sessions


# ============================================================
# EEGNet 模型
# ============================================================
class EEGNet(nn.Module):
    """EEGNet (Lawhern et al., 2018) 实现。

    输入: [batch, 1, channels, timepoints]
    输出: [batch, n_classes]
    """
    def __init__(self, n_channels=58, n_timepoints=1000, n_classes=2,
                 F1=8, D=2, F2=16, dropout=0.5, kernel_length=64):
        super().__init__()
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints

        # Block 1: Temporal Convolution + Depthwise Spatial Convolution
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.conv2 = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu1 = nn.ELU()
        self.avgpool1 = nn.Avg2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)

        # Block 2: Separable Convolution
        self.conv3 = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8),
                               groups=F1 * D, bias=False)
        self.conv4 = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.avgpool2 = nn.Avg2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)

        # 分类器
        # 计算展平后的特征维度
        self._flat_size = self._get_flat_size()
        self.classifier = nn.Linear(self._flat_size, n_classes)

    def _get_flat_size(self):
        """计算卷积后的特征维度。"""
        x = torch.zeros(1, 1, self.n_channels, self.n_timepoints)
        x = self.avgpool1(self.elu1(self.bn2(self.conv2(self.bn1(self.conv1(x))))))
        x = self.dropout1(x)
        x = self.avgpool2(self.elu2(self.bn3(self.conv4(self.conv3(x)))))
        x = self.dropout2(x)
        return x.view(1, -1).shape[1]

    def forward(self, x):
        # x: [batch, channels, timepoints] → [batch, 1, channels, timepoints]
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)

        # 分类
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ============================================================
# 训练与评估
# ============================================================
def train_model(model, train_loader, criterion, optimizer, n_epochs=100, device="cpu"):
    """训练模型。"""
    model.train()
    for epoch in range(n_epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
    return model


def evaluate_model(model, test_loader, device="cpu"):
    """评估模型，返回准确率。"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            _, predicted = torch.max(output, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    return correct / total if total > 0 else 0.0


def make_loader(X, y, batch_size=16, shuffle=True):
    """创建 DataLoader。"""
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ============================================================
# 评估协议
# ============================================================
def protocol_within_session(sessions, output_dir, n_folds=10, n_epochs=100, device="cpu"):
    """Within-session 10-fold CV 评估。"""
    print("\n" + "=" * 60)
    print("协议: Within-Session 10-fold CV")
    print("=" * 60)

    results = []
    subjects = defaultdict(dict)
    for (sub_id, ses_id), data in sessions.items():
        subjects[sub_id][ses_id] = data

    total = sum(len(v) for v in subjects.values())
    done = 0

    for sub_id in sorted(subjects.keys()):
        for ses_id in sorted(subjects[sub_id].keys()):
            done += 1
            X = subjects[sub_id][ses_id]["X"]
            y = subjects[sub_id][ses_id]["y"]

            # 标签重映射为 0-indexed
            y_unique = np.unique(y)
            y_map = {v: i for i, v in enumerate(y_unique)}
            y_mapped = np.array([y_map[v] for v in y])

            # 10-fold CV
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            fold_accs = []

            for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y_mapped[train_idx], y_mapped[test_idx]

                model = EEGNet(n_channels=X.shape[1], n_timepoints=X.shape[2],
                               n_classes=len(y_unique)).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                train_loader = make_loader(X_train, y_train, batch_size=16)
                test_loader = make_loader(X_test, y_test, batch_size=16, shuffle=False)

                model = train_model(model, train_loader, criterion, optimizer,
                                    n_epochs=n_epochs, device=device)
                acc = evaluate_model(model, test_loader, device=device)
                fold_accs.append(acc)

            mean_acc = np.mean(fold_accs)
            std_acc = np.std(fold_accs)
            results.append({
                "subject": sub_id, "session": ses_id,
                "accuracy": mean_acc, "std": std_acc,
                "protocol": "within_session"
            })
            print(f"  [{done}/{total}] {sub_id}/{ses_id}: "
                  f"{mean_acc:.4f} ± {std_acc:.4f}")

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "results_within_session.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n保存至: {csv_path}")
    return df


def protocol_cross_session(sessions, output_dir, n_epochs=100, device="cpu"):
    """Cross-session 评估：train on ses-i, test on ses-j。"""
    print("\n" + "=" * 60)
    print("协议: Cross-Session")
    print("=" * 60)

    results = []
    subjects = defaultdict(dict)
    for (sub_id, ses_id), data in sessions.items():
        subjects[sub_id][ses_id] = data

    # 只处理有 3 个 session 的被试
    valid_subjects = {k: v for k, v in subjects.items() if len(v) >= 2}
    total_pairs = sum(len(v) * (len(v) - 1) // 2 for v in valid_subjects.values())
    done = 0

    for sub_id in sorted(valid_subjects.keys()):
        ses_dict = valid_subjects[sub_id]
        ses_ids = sorted(ses_dict.keys())

        for i in range(len(ses_ids)):
            for j in range(i + 1, len(ses_ids)):
                done += 1
                ses_i, ses_j = ses_ids[i], ses_ids[j]
                Xi, yi = ses_dict[ses_i]["X"], ses_dict[ses_i]["y"]
                Xj, yj = ses_dict[ses_j]["X"], ses_dict[ses_j]["y"]

                # 标签重映射
                y_unique = np.unique(np.concatenate([yi, yj]))
                y_map = {v: k for k, v in enumerate(y_unique)}
                yi_mapped = np.array([y_map[v] for v in yi])
                yj_mapped = np.array([y_map[v] for v in yj])

                # 正向: train on i, test on j
                model = EEGNet(n_channels=Xi.shape[1], n_timepoints=Xi.shape[2],
                               n_classes=len(y_unique)).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                train_loader = make_loader(Xi, yi_mapped, batch_size=16)
                test_loader = make_loader(Xj, yj_mapped, batch_size=16, shuffle=False)

                model = train_model(model, train_loader, criterion, optimizer,
                                    n_epochs=n_epochs, device=device)
                acc_ij = evaluate_model(model, test_loader, device=device)

                # 反向: train on j, test on i
                model = EEGNet(n_channels=Xi.shape[1], n_timepoints=Xi.shape[2],
                               n_classes=len(y_unique)).to(device)
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                train_loader = make_loader(Xj, yj_mapped, batch_size=16)
                test_loader = make_loader(Xi, yi_mapped, batch_size=16, shuffle=False)

                model = train_model(model, train_loader, criterion, optimizer,
                                    n_epochs=n_epochs, device=device)
                acc_ji = evaluate_model(model, test_loader, device=device)

                results.append({
                    "subject": sub_id, "train_ses": ses_i, "test_ses": ses_j,
                    "accuracy": acc_ij, "protocol": "cross_session"
                })
                results.append({
                    "subject": sub_id, "train_ses": ses_j, "test_ses": ses_i,
                    "accuracy": acc_ji, "protocol": "cross_session"
                })
                print(f"  [{done}/{total_pairs}] {sub_id}: "
                      f"{ses_i}→{ses_j} = {acc_ij:.4f}, "
                      f"{ses_j}→{ses_i} = {acc_ji:.4f}")

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "results_cross_session.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n保存至: {csv_path}")
    return df


def protocol_loso(sessions, output_dir, n_epochs=100, device="cpu"):
    """LOSO (Leave-One-Subject-Out) 跨被试评估。"""
    print("\n" + "=" * 60)
    print("协议: Leave-One-Subject-Out (LOSO)")
    print("=" * 60)

    results = []
    subjects = defaultdict(dict)
    for (sub_id, ses_id), data in sessions.items():
        subjects[sub_id][ses_id] = data

    all_sub_ids = sorted(subjects.keys())
    n_subjects = len(all_sub_ids)

    for idx, test_sub in enumerate(all_sub_ids):
        # 训练集: 除 test_sub 外的所有被试的所有 session
        X_train_list, y_train_list = [], []
        for sub_id in all_sub_ids:
            if sub_id == test_sub:
                continue
            for ses_id in sorted(subjects[sub_id].keys()):
                X_train_list.append(subjects[sub_id][ses_id]["X"])
                y_train_list.append(subjects[sub_id][ses_id]["y"])

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        # 测试集: test_sub 的所有 session
        y_unique = np.unique(y_train)
        y_map = {v: k for k, v in enumerate(y_unique)}
        y_train_mapped = np.array([y_map[v] for v in y_train])

        model = EEGNet(n_channels=X_train.shape[1], n_timepoints=X_train.shape[2],
                       n_classes=len(y_unique)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_loader = make_loader(X_train, y_train_mapped, batch_size=16)
        model = train_model(model, train_loader, criterion, optimizer,
                            n_epochs=n_epochs, device=device)

        # 逐 session 测试
        for ses_id in sorted(subjects[test_sub].keys()):
            X_test = subjects[test_sub][ses_id]["X"]
            y_test = subjects[test_sub][ses_id]["y"]
            y_test_mapped = np.array([y_map[v] for v in y_test])

            test_loader = make_loader(X_test, y_test_mapped, batch_size=16, shuffle=False)
            acc = evaluate_model(model, test_loader, device=device)

            results.append({
                "subject": test_sub, "session": ses_id,
                "accuracy": acc, "protocol": "loso"
            })

        # 打印该被试的平均准确率
        sub_accs = [r["accuracy"] for r in results if r["subject"] == test_sub]
        print(f"  [{idx + 1}/{n_subjects}] {test_sub}: "
              f"mean acc = {np.mean(sub_accs):.4f}")

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "results_loso.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n保存至: {csv_path}")
    return df


# ============================================================
# 可视化
# ============================================================
def generate_figures(results_dict, output_dir):
    """生成结果可视化。"""
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # --- 图1: Within-session 准确率分布 ---
    if "within" in results_dict:
        df = results_dict["within"]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=df, x="subject", y="accuracy", ax=ax, color="steelblue")
        ax.axhline(df["accuracy"].mean(), color="red", linestyle="--",
                   label=f'均值 = {df["accuracy"].mean():.3f}')
        ax.set_xlabel("Subject")
        ax.set_ylabel("Accuracy")
        ax.set_title("Within-Session 10-fold CV 准确率")
        ax.legend()
        plt.xticks(rotation=90, fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "within_session_accuracy.png"), dpi=150)
        plt.close()
        print("  [图1] within_session_accuracy.png")

    # --- 图2: Cross-session 准确率矩阵 ---
    if "cross" in results_dict:
        df = results_dict["cross"]
        # 构建 train_ses → test_ses 的准确率矩阵
        pivot = df.pivot_table(index="train_ses", columns="test_ses",
                               values="accuracy", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".3f", cmap="YlOrRd",
                    vmin=0.5, vmax=1.0)
        ax.set_xlabel("Test Session")
        ax.set_ylabel("Train Session")
        ax.set_title("Cross-Session 准确率矩阵（所有被试平均）")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "cross_session_matrix.png"), dpi=150)
        plt.close()
        print("  [图2] cross_session_matrix.png")

        # 每个被试的 cross-session 准确率
        fig, ax = plt.subplots(figsize=(10, 5))
        sub_means = df.groupby("subject")["accuracy"].mean().sort_index()
        sub_means.plot(kind="bar", ax=ax, color="steelblue")
        ax.axhline(sub_means.mean(), color="red", linestyle="--",
                   label=f'均值 = {sub_means.mean():.3f}')
        ax.set_xlabel("Subject")
        ax.set_ylabel("Accuracy")
        ax.set_title("Cross-Session 准确率（每个被试平均）")
        ax.legend()
        plt.xticks(rotation=90, fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "cross_session_per_subject.png"), dpi=150)
        plt.close()
        print("  [图3] cross_session_per_subject.png")

    # --- 图3: LOSO 准确率分布 ---
    if "loso" in results_dict:
        df = results_dict["loso"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 每个被试的平均 LOSO 准确率
        sub_means = df.groupby("subject")["accuracy"].mean().sort_index()
        axes[0].bar(range(len(sub_means)), sub_means.values, color="steelblue")
        axes[0].axhline(sub_means.mean(), color="red", linestyle="--",
                        label=f'均值 = {sub_means.mean():.3f}')
        axes[0].set_xticks(range(len(sub_means)))
        axes[0].set_xticklabels(sub_means.index, rotation=90, fontsize=7)
        axes[0].set_xlabel("Subject")
        axes[0].set_ylabel("LOSO Accuracy")
        axes[0].set_title("LOSO 跨被试准确率")
        axes[0].legend()

        # 准确率分布直方图
        axes[1].hist(sub_means.values, bins=20, edgecolor="black", alpha=0.7)
        axes[1].axvline(sub_means.mean(), color="red", linestyle="--",
                        label=f'均值 = {sub_means.mean():.3f}')
        axes[1].axvline(0.5, color="gray", linestyle=":", label="随机水平")
        axes[1].set_xlabel("LOSO Accuracy")
        axes[1].set_ylabel("Subject 数量")
        axes[1].set_title("LOSO 准确率分布")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "loso_accuracy.png"), dpi=150)
        plt.close()
        print("  [图4] loso_accuracy.png")

    # --- 图4: 三种协议对比 ---
    summary = {}
    for key, df in results_dict.items():
        summary[key] = {
            "mean": df["accuracy"].mean(),
            "std": df["accuracy"].std(),
            "n": len(df)
        }

    if len(summary) >= 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        names = list(summary.keys())
        means = [summary[k]["mean"] for k in names]
        stds = [summary[k]["std"] for k in names]
        colors = ["steelblue", "coral", "seagreen"][:len(names)]

        bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
        ax.axhline(0.5, color="gray", linestyle=":", label="随机水平")
        ax.set_ylabel("Accuracy")
        ax.set_title("三种评估协议准确率对比")
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{m:.3f}", ha="center", fontsize=10)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "protocol_comparison.png"), dpi=150)
        plt.close()
        print("  [图5] protocol_comparison.png")

    print(f"\n所有图表已保存至: {fig_dir}")


# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEGNet Cross-Session Baseline")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="预处理 .npz 文件目录")
    parser.add_argument("--output_dir", type=str, default="./baseline_results",
                        help="输出目录")
    parser.add_argument("--protocol", type=str, default="all",
                        choices=["within", "cross", "loso", "all"],
                        help="评估协议")
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="训练 epoch 数")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size")
    parser.add_argument("--device", type=str, default="auto",
                        help="设备 (cpu/cuda/auto)")
    args = parser.parse_args()

    # 设备选择
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"设备: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据
    sessions = load_sessions(args.data_dir)

    # 运行评估
    results_dict = {}
    if args.protocol in ("within", "all"):
        results_dict["within"] = protocol_within_session(
            sessions, args.output_dir, n_epochs=args.n_epochs, device=device)
    if args.protocol in ("cross", "all"):
        results_dict["cross"] = protocol_cross_session(
            sessions, args.output_dir, n_epochs=args.n_epochs, device=device)
    if args.protocol in ("loso", "all"):
        results_dict["loso"] = protocol_loso(
            sessions, args.output_dir, n_epochs=args.n_epochs, device=device)

    # 生成可视化
    if results_dict:
        generate_figures(results_dict, args.output_dir)

    print("\n全部完成！")
