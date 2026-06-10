"""数据子包：Dataset / DataLoader / 按被试划分。

关键约定：
  - Dataset 返回 X=[channels, time]，y 为标量。
  - 划分必须以被试为单位（subject-wise），绝不按 trial 划分。
主模块：shu_dataset（数据集与划分工具）。
"""
