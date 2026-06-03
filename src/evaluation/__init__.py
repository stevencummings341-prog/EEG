"""评估子包：分类指标 + 置信度校准指标 + 曲线。

分类：accuracy / balanced accuracy / macro-F1 / AUC。
校准：ECE / NLL / Brier；曲线：risk-coverage、performance-vs-trial。
见 docs/EXPERIMENT_PROTOCOL.md。
"""

from . import metrics  # noqa: F401
