"""分类与置信度校准指标。

分类指标用 sklearn 实现（稳定、无需自造轮子）。校准指标 (ECE/NLL/Brier)
自实现，便于在线/离线统一调用。所有函数输入 numpy 数组。
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """返回 accuracy / balanced_accuracy / macro_f1。"""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
    )

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def auc_binary(y_true: np.ndarray, prob_pos: np.ndarray) -> float:
    """二分类 AUC（prob_pos 为正类概率）。退化情形返回 nan。"""
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true).ravel()
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, np.asarray(prob_pos).ravel()))


def auc_multiclass(y_true: np.ndarray, probs: np.ndarray) -> float:
    """多分类 macro-OVR AUC。类别不足或 sklearn 失败时返回 nan。"""
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true).ravel()
    probs = np.asarray(probs)
    if probs.ndim != 2 or probs.shape[0] != y_true.size:
        return float("nan")
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def expected_calibration_error(y_true, probs, n_bins: int = 15) -> float:
    """ECE：把样本按预测置信度分箱，统计 |置信度 - 准确率| 的加权平均。

    probs: [N, C] 概率。置信度取每个样本的最大概率。
    """
    y_true = np.asarray(y_true).ravel()
    probs = np.asarray(probs)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        # 最后一个箱包含右端点 1.0
        in_bin = (conf > lo) & (conf <= hi) if hi < 1.0 else (conf > lo) & (conf <= hi + 1e-9)
        m = in_bin.sum()
        if m == 0:
            continue
        ece += (m / n) * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def negative_log_likelihood(y_true, probs, eps: float = 1e-12) -> float:
    """NLL（多分类交叉熵，自然对数）。"""
    y_true = np.asarray(y_true).ravel().astype(int)
    probs = np.clip(np.asarray(probs), eps, 1.0)
    return float(-np.log(probs[np.arange(len(y_true)), y_true]).mean())


def brier_score(y_true, probs) -> float:
    """多分类 Brier 分数：预测概率与 one-hot 标签的均方误差（对类求和）。"""
    y_true = np.asarray(y_true).ravel().astype(int)
    probs = np.asarray(probs)
    n, c = probs.shape
    onehot = np.zeros((n, c), dtype=np.float64)
    onehot[np.arange(n), y_true] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())
