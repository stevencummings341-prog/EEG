# eog_ecg_clean 预处理数据质量总结（中文）

> 数据：`/share/workspace2/moto_imagination/WBCIC_SHU/processed/eog_ecg_clean/`（每个 session 一个 `.npz`）
> 对照：官方论文 `derivatives/2C dataset_processeddata`（每个 session 一个 `.mat`）
> 依据：`scripts/compare_processed_quality.py` 全量质检 + 对比的结果
> （`manifest_qc_summary.json` / `session_quality_metrics.csv` / `paired_similarity_metrics.csv` / `session_alignment.json` / `figures/`）

---

## 0. 一句话结论

我们的 `eog_ecg_clean` 预处理**质量合格、可用于训练**：形状/标签/采样率全部正确、无 NaN/Inf；
与官方 derivatives 在频谱、幅值、MI 判别信息上高度一致；额外的 EOG/ECG 辅助 ICA 清理确实
在有伪迹时去掉了眼电/心电成分、又没有破坏运动想象信息。**建议用 status=ok 的 148 个 session
作为 `eog_ecg_clean_v1` 训练入口，先排除 5 个 failed session。**

---

## 1. 整体数据质量如何

全部 **153** 个 session（51 被试 × 3 session）逐项检查（`manifest_qc_summary.json`）：

| 检查项                    | 结果                        |
| ---------------------- | ------------------------- |
| 总 session 数            | 153（符合预期）                 |
| status=ok              | **148 / 153**（5 个 failed） |
| X 形状 [200,58,1000]     | 148 个通过（5 个 failed 因试次不足） |
| y 形状 [200]             | 同上                        |
| 标签 100/100             | 148 个通过                   |
| trigger 数 = 200        | 148 个通过                   |
| 采样率 = 250 Hz           | **153/153 全通过**           |
| 通道数 = 58               | **153/153 全通过**           |
| 时间点 = 1000             | **153/153 全通过**           |
| NaN / Inf              | **0（完全没有）**               |
| 标签多重集不匹配（危险信号）         | **0**                     |
| 辅助降噪 aux_cleaning_used | **153/153**               |
| 总输出大小                  | 6.15 GiB                  |

> 结论：除了 5 个因原始触发缺失而试次不足的 session，其余 148 个在“结构正确性”上**全部满分**，
> 数据干净（无 NaN/Inf）、量纲正确（µV）、左右手 100/100 平衡。

---

## 2. “148 ok / 5 failed” 是什么意思

我们的预处理脚本对每个 session 有一套**硬质检判定**（`evaluate_failure_reasons`）：

- **ok**：形状 = [200,58,1000]、标签 = 100/100、trigger = 200、无 NaN/Inf（且有 .mat 时多重集匹配）。
- **failed**：上面任意一条不满足。

“148 ok / 5 failed”即：**148 个 session 完全合规、可直接进入训练；5 个 session 因试次数不足 200
被标记为 failed**。failed 不是“数据坏了”，而是“这个 session 没凑齐 200 个完整试次”，我们选择
显式标记而不是悄悄丢弃，方便后续决定排除还是回原始数据补救。

---

## 3. 5 个 failed session 及失败原因

| session          | 实际试次 | 左/右标签    | trigger 数 | 失败原因                     |
| ---------------- | ---- | -------- | --------- | ------------------------ |
| `sub-023/ses-01` | 199  | 99 / 100 | 199       | 少 1 个试次（trigger 199≠200） |
| `sub-024/ses-02` | 199  | 99 / 100 | 199       | 少 1 个试次                  |
| `sub-024/ses-03` | 195  | 99 / 96  | 195       | 少 5 个试次                  |
| `sub-026/ses-01` | 199  | 99 / 100 | 199       | 少 1 个试次                  |
| `sub-032/ses-02` | 199  | 99 / 100 | 199       | 少 1 个试次                  |

> 共同点：都是 **trigger/试次数 < 200**，导致形状不是 [200,58,1000]、标签不是 100/100。

---

## 4. 为什么 failed 不是“降噪错误”，而是 raw trigger/试次缺失

1. **失败发生在“切试次”这一步，而不是 ICA 清理这一步。** 失败原因全部是 `trigger_count = 199/195`，
   即从原始 `evt.bdf` 解析到的运动想象事件本来就少于 200 个。EEG 波形本身没有 NaN/Inf、没有崩坏。
2. **这 5 个 session 的辅助降噪是正常完成的**（`aux_cleaning_used=True`，ICA 正常拟合、正常排除分量），
   说明 EOG/ECG 清理流程没有报错。失败与降噪无关。
3. **同一被试的其它 session 正常**：例如 sub-024 的 ses-01 是 ok 的，sub-023 的 ses-02/03 也是 ok 的。
   如果是降噪代码的系统性 bug，应该是整个被试或大面积 session 全挂，而不是零星几个少 1~5 个试次。
4. **本质是采集端/触发端问题**：原始记录里这几段就缺了 1~5 个触发标记（漏触发或被试少做了几个 trial）。

> 一句话：**“缺试次”是原始数据的事实，不是我们清理把数据搞坏了。**

---

## 5. 和官方 derivatives 的对比结果（总览）

在 148 个可比的 ok session 上（5 个 failed 因试次数不同无法逐试次配对），核心指标：

| 指标                                       | 数值        | 含义                |
| ---------------------------------------- | --------- | ----------------- |
| 全局 std 比值（ours/official）中位数              | **0.976** | 整体幅值与官方几乎一致，略低    |
| 全局 RMS 比值中位数                             | **0.976** | 同上                |
| μ/α 带功率比值（8–13 Hz）中位数                    | **0.898** | MI 关键频段功率略低（清理所致） |
| β 带功率比值（13–30 Hz）中位数                     | **0.941** | β 段几乎一致           |
| 配对 trial-wise 相关中位数（144 个 exact session） | **0.954** | 单试次波形与官方高度一致      |
| 配对 relative RMSE 中位数                     | **0.251** | 残差主要来自滤波实现差异，量级合理 |
| 多重集标签不匹配                                 | **0**     | 标签完全对得上           |

> 结论：**对比通过**。我们的数据在“频谱形状、幅值水平、单试次波形、标签”四个维度上都与官方
> paper-style 数据高度一致；差异方向（ours 略低）与“我们多做了一步辅助降噪”完全吻合。

---

## 6. std/RMS ratio 0.976、trial-wise corr 0.954、mu/beta ratio 怎么解读

- **std / RMS ratio ≈ 0.976（≈1，略小于 1）**
  比值=我们的幅值 ÷ 官方幅值。0.976 表示整体幅值水平和官方几乎相同，只低约 2.4%。
  153 个中有 111 个 ours 更低——这正是**眼电/心电伪迹被 ICA 清掉后总能量略降**的预期表现。
  不是数据被削弱或缩放错误（如果是量纲/缩放 bug，比值会是 ~1e6 或 ~0.001 这种离谱值）。

- **trial-wise corr ≈ 0.954（很高）**
  对“试次顺序能精确对齐”的 144 个 session，把我们的单个 trial（58×1000 展平）和官方对应 trial
  算 Pearson 相关，中位数 0.954。**说明逐个试次的波形与官方几乎是同一条曲线**，预处理主干
  （去辅助通道→Pz 重参考→0.5–40 带通+50 notch→[0,4)s 切窗→250 Hz）与官方一致。
  没到 1.0 是因为：(a) 我们额外做了 ICA 去伪迹；(b) MNE(firwin) 与 EEGLAB 的滤波实现略有差异。

- **μ/β bandpower ratio（0.898 / 0.941）**
  μ/α（8–13 Hz）和 β（13–30 Hz）是运动想象最关键的频段。比值接近 1（β 更接近）说明
  **MI 相关的节律功率被完整保留**；μ 段略低是因为眼动等低频伪迹常落在更低频，被清理时连带影响很小一部分。
  这是“清理有效但没伤到 MI 信号”的理想结果。

---

## 7. EOG/ECG 辅助清理是否有效

**有效，且“按需触发”**（来自 `manifest_qc_summary.json` 的统计）：

- **EOG（眼电）**：**106 / 153** 个 session 检出并剔除了眼电相关独立成分，共 **189** 个分量
  （每 session 0–5 个，均值 1.24）。说明大多数 session 确有眼动污染并被清掉。
- **ECG（心电）**：**43 / 153** 个 session 剔除了心电相关分量，共 **74** 个。心电污染本就比眼电少，
  只在真有心电成分（相关性超阈值）时才剔除，**没有过度清理**。
- **稳健性**：高幅 session 上 `n_components=0.99` 会塌缩，脚本自动用固定分量数重试，
  共触发 **10** 次；**no-aux-clean 回退 0 次**（即没有任何 session 因 ICA 失败而退化为不清理）。

> 综合第 6 节：清理后幅值/μ-β 功率略降（去掉了非脑成分），而单试次相关仍 0.954、MI 可分性保留
> （见第 9 节图），说明 **EOG/ECG 清理“去伪迹、留信号”，达到了预期目的**。

---

## 8. 官方 derivatives 的 session 顺序错位问题（重要）

对比时发现一个**反直觉但已被证实**的现象：**官方 derivatives 把部分被试的 session 存成了与
BIDS sourcedata（也就是我们的 ses-YY）不同的顺序**。

- **现象**：若直接“同名 ses 对同名 ses”比较，部分被试会出现荒唐的 std 比值（如 sub-030 的 ses-01
  比值 264×，sub-018 的 ses-01 比值 14×，而同一被试另一个 session 又 < 0.1）——一高一低成对出现，
  正是“两个 session 被互换”的特征。
- **处理**：脚本在**每个被试内部**用对 ICA 稳健的幅值指纹 (std, max|·|) 把“我们的每个 session”
  匹配到“官方真正对应的那个 session”（只有当某个排列明显优于同序时才判为错位，避免误配）。
- **交叉验证**：对齐后，这些被试的标签从“顺序不一致”变为**精确一致**、单试次相关大幅回升
  （例如 sub-001/ses-02 在同名比较下只有 102/200，对齐后发现它其实对应官方 ses-03，标签精确匹配、
  相关 0.92）。波形 overlay 也几乎重合。这说明**匹配是对的**。
- **受影响被试：22 / 51**：sub-001, 004, 006, 007, 008, 010, 013, 017, 018, 023, 025, 027, 030,
  036, 038, 039, 042, 044, 045, 047, 049, 051。
- **结论**：这**不是我们预处理的错误**。我们的 (X, y) 严格来自 sourcedata + evt.bdf，被试内 session
  的幅值集合与官方完全一致，只是官方 `.mat` 的 ses 命名顺序不同。
  （此前记录里“试次顺序不一致”的现象，现在看大多其实是 session 互换。）
  报告中的所有比值/相关都是**对齐之后**算的，因此那些虚假异常已经消失。

---

## 9. 哪些图最适合汇报

全部 13 张图在 `qc_vs_derivatives/figures/`。汇报/写文档时**优先用这几张**：

1. `figures/psd_overlay_C3_C4_Cz.png` —— **最有说服力**：C3/C4/Cz 上 ours 与 official 的功率谱在
   μ(8–13)/β(13–30) 频段几乎完全重合，只有 40–50 Hz 因滤波实现+50 Hz 陷波不同而分开（高于 MI 频段，不影响）。
2. `figures/std_ours_vs_official_scatter.png` —— 每个 session 的 std 散点紧贴 y=x（含极端高幅 session），
   直观说明“对齐后幅值与官方一致”。
3. `figures/class_mu_beta_difference_C3_C4.png` —— C3/C4 左右手 μ/β 可分性（|Cohen's d|）ours≈official，
   证明 **MI 判别信息被保留**。
4. `figures/qc_dashboard.png` —— 一张总览：148 ok / 5 failed、144 exact、aux 153、ICA 回退 10、
   无 NaN/Inf、22 个被试官方 ses 错位。
5. `figures/example_waveform_overlay.png` —— 单试次波形 ours vs official 几乎重合（含被互换又被对齐回来的 session），最直观。
6. `figures/bandpower_ratio_mu_beta.png` —— μ/β 带功率比值分布集中在 ~0.9–1.0。

> 辅助/补充：`std_ratio_hist.png`、`rms_boxplot_ours_vs_official.png`、`channel_rms_ratio_heatmap.png`、
> `trial_corr_hist_exact_label_sessions.png`、`ica_excluded_components_summary.png`、
> `high_amp_trial_ratio_boxplot.png`、`psd_overlay_global.png`。

### MI 可分性对照表（来自报告）

| 通道/频带 | ours \|Cohen d\| | official \|Cohen d\| |
| --- | --- | --- |
| C3 μ | 0.138 | 0.150 |
| C3 β | 0.112 | 0.112 |
| C4 μ | 0.155 | 0.161 |
| C4 β | 0.142 | 0.144 |

> ours 与 official 量级一致 → 清理没有破坏左右手区分信息。

---

## 10. 最终结论与建议

- **数据质量：合格、可用于训练。** 结构正确、无 NaN/Inf、量纲正确；与官方 derivatives 频谱/幅值/
  单试次/标签一致；EOG/ECG 清理有效且不伤 MI 信号。
- **建议训练入口：使用 status=ok 的 148 个 session 作为 `eog_ecg_clean_v1`。**
  代码层面 `SHUTrialDataset.from_manifest(..., statuses=('ok',))` 默认即过滤 failed。
- **暂时排除 5 个 failed session**（`sub-023/ses-01`、`sub-024/ses-02`、`sub-024/ses-03`、
  `sub-026/ses-01`、`sub-032/ses-02`）。它们的被试仍有其它正常 session，**不影响被试级 41/10 划分**。
  后续若想找回，可回原始 `evt.bdf` 重新解析触发（属采集端问题，与降噪无关）。
- **官方 ses 错位**只影响“与官方对照”，不影响我们自己的训练（我们用自己的 sourcedata-对齐数据 + evt.bdf 标签）。

> 下一步（待确认）：进入 41/10 被试级划分 + `SHUTrialDataset`。本总结不启动任何训练。
