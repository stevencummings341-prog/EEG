# P10_INTEGRATION_SUMMARY.md

> 学长 P10 文件夹与主项目的对照说明。目的：防止以后混乱——哪个是参考、哪个是主项目、哪些结果是同一回事。

## 1. P10 来源

- 外部路径：`/share/home/yuan/SYX/P10_MI泛化研究/`
- 它是**参考材料 / 方向来源**，**不是主运行目录**。
- 主项目仍是 `/share/home/yuan/SYX/eeg-mi-online/`。

## 2. P10 与主项目的关系

- P10 **Phase 0 drift** = 主项目 `outputs/analysis/session_drift_v1/`。
- P10 **Phase 1 baseline** = 主项目 `outputs/experiments/baseline_v1/provenance/session_model_compare_v1/`。
- 数字一致，是**同一研究的不同整理版本**（P10 是对外交接版）。
- P10 **Phase 2 online code** 是**草稿 / 设计，未在服务器验证** → 主项目归为 future。

## 3. 已整合事实

- **漂移诊断完成**：跨 session 漂移以**空间模式 + μ/β 频谱**为主（CSP≈0.420、ERD-μ≈0.419、μ-KS≈0.246），
  幅值稳定（RMS≈0.992），可分性无系统变化（Fisher≈0）。
- **baseline 完成**：**within > cross**，cross-session drop 约 **9–13%**。
- **方向不对称**：`ses-02→03` 最好（EEGNet 0.749），`ses-03→01` 最差（0.681），差 6.8pp。
- **MMD 与 cross accuracy 反向相关**（按 pair Spearman ρ=−1.0，n=3）。
- **Step 1 multi-source 已补**：`ses-01+02 → ses-03`，三模型均优于最强单源（EEGNet 0.7717，+0.0224）。

## 4. 当前主线

- **A** drift ✅
- **B** static baseline（within + single-source cross）✅
- **C** multi-source Step 1（ses-01+02→ses-03）✅
- **D** Step 2 no-learning adaptation ⏳（下一步，未运行）
- online / 41-10 / agent / prototype / memory / CAP-EEGNet full = **future**

## 5. P10 code 处理原则

- **不直接覆盖主项目代码**。
- 作为**参考**。
- 未来若迁移，要进入 `src/` 和 `scripts/` 的规范位置（落点见 `docs/CODE_INTEGRATION_NOTES.md`）。

> 运行假设差异：P10 示例在 `/share/workspace2/.../WBCIC_SHU/` 下直接跑/写；主项目**不**这么做
> （数据只读、不写 workspace2、路径走 `configs/paths.yaml`）。**采纳 P10 方法学，不采纳其运行路径假设。**

---

本文件意义：把 P10 文件夹变成**可追溯参考**，不要变成第二套混乱项目。
外部文件清单见 `docs/references/P10_MI_generalization_README.md`。
