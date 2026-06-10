"""在线学习子包：test-then-update（prequential）主循环与轻量更新模块。

铁律（30-experiment-protocol 规则）：每个 trial 先预测+记录，再更新。
只更新 prototype / adapter / calibration head / BN 统计 / （有标签时）分类头；
默认冻结 backbone。无标签时按 confidence 阈值门控。
"""
