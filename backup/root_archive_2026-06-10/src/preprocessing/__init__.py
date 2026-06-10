"""预处理子包：raw BDF -> [trials, channels, time] 处理后张量。

两种变体（见 docs/PREPROCESSING_SPEC.md）：
  - paper_style：先实现，不做 ICA。
  - eog_ecg_clean：第二阶段，用 ECG/EOG 辅助去伪迹。
主入口：shu_preprocess.preprocess_one_session
"""
