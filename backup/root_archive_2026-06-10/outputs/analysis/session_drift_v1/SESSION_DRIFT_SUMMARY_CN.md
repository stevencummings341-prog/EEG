# 跨 Session 漂移诊断 — 一页中文总结

## 做了什么
- 量化同一被试在不同 session（不同天）之间的 EEG 分布漂移，回答“跨 session 泛化为什么难”。
- 纯数据层面的诊断，**不是模型训练**，不涉及 train/test、不产生 accuracy。

## 怎么做的
- 数据：`eog_ecg_clean_v1` 的 **148 个 status=ok session**（已排除 5 个 failed）。
- 每个被试 3 个 session（ses-01/02/03），按被试内部 3 种 pair（1-2、1-3、2-3）算漂移。
- 因 failed session，最终 **144 pairs / 50 被试**（sub-024 只剩 1 个 ok session → 0 pair）。
- 指标：MMD、CORAL、μ/β 功率漂移、μ-KS、ERD/ERS 空间相关、CSP 相似度、RMS 比值、Fisher 可分性。

## 三个 session pair 的结果
| pair | n | MMD(mean) | μ-KS | CSP相似 | ERD μ corr |
|---|---|---|---|---|---|
| 01-02 | 47 | 0.237 | 0.275 | 0.411 | 0.384 |
| 01-03 | 48 | 0.247 | 0.270 | 0.421 | 0.414 |
| 02-03 | 49 | 0.230 | 0.196 | 0.427 | 0.459 |

- **整体分布距离(MMD)最大**的 pair：`01-03`（相隔最远的 1-3）；而 **02-03** 在 μ-KS/CSP/ERD 上最稳定，提示后期 session 的空间-频谱模式趋于一致（部分支持学习效应，详见完整报告 E 节）。

## 每个被试怎么看
- 看 `per_subject_drift_summary.csv` / `.md`：每被试一行，含各指标均值与 `drift_level`。
- high-drift 被试 17 个、stable 17 个；热力图见 `figures/subject_*_heatmap.png`。

## 主要结论
- 跨 session 漂移主要体现在**空间模式 + μ/β 频谱分布**（CSP≈0.420、ERD μ≈0.419、μ-KS≈0.246）。
- 幅值不是主因（RMS 比值中位数≈0.992≈1）。
- Fisher shift 平均≈-0.001≈0：左右手可分性没有统一增强或减弱，性能下降更可能来自模式漂移而非类别不可分。

## 下一步怎么接 baseline
- 用同一批 148 个 ok session 跑 within-session CV vs cross-session，量化 accuracy drop。
- 自适应方法优先考虑 spatial alignment（EA/CORAL）、frequency/filter-bank adaptation、BN/adapter，而不是只做全局幅值归一化。
