"""可视化子包：数据质检 / 与官方 derivatives 对比的图表。

所有绘图用非交互后端（Agg），适合在计算节点无显示环境下出图。
见 src/visualization/quality_plots.py 与 scripts/compare_processed_quality.py。
"""

from . import quality_plots  # noqa: F401
