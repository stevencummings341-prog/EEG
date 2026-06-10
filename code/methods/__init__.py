"""No-learning / unsupervised test-time alignment baselines (Step 2).

This package implements the Step-2 cross-session adaptation baselines that DO NOT
train on target labels. The fairness boundary is strict (see
``docs/ADAPTATION_BASELINE_PLAN.md``):

  * the model is trained normally on the SOURCE session(s);
  * the TARGET/test session may only be used through its UNLABELED ``X`` to
    estimate alignment statistics (z-score mean/std, covariance reference, BN
    running stats, band-power profile);
  * ``y_test`` is used ONLY for the final evaluation — never for training,
    validation, early stopping, hyper-parameter selection, or method selection;
  * no ``optimizer.step`` is ever taken on the target (only BN running-stat
    updates are allowed, and only for the ``bn_statistics_adaptation`` method).

Modules:
  * ``session_alignment`` — feature-space alignment transforms (channel z-score,
    Euclidean Alignment, Riemannian Alignment, filter-bank reweighting) + a
    registry. All fit source statistics from the SOURCE TRAIN split only and
    target statistics from the TARGET X only.
  * ``bn_adaptation`` — BatchNorm running-statistic adaptation (forward target X,
    update only BN running mean/var; no loss, no backward, no optimizer step).
"""

from .session_alignment import (  # noqa: F401
    ALIGNMENT_METHODS,
    ChannelZScore,
    EuclideanAlignment,
    FilterBankReweight,
    RiemannianAlignment,
    make_alignment_method,
)
from .bn_adaptation import adapt_bn_statistics, count_bn_layers  # noqa: F401
