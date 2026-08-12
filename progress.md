# PROGRESS / Memory Log

The project's running memory. **Append a dated entry whenever you finish a
meaningful step or make a design decision.** Newest entries on top. Keep it factual.

Format per entry: date, what was done, decisions made, open questions, next step.

---

## 2026-08-12 — `paper_baseline_3c_821_v1` 续跑重提（QOS 墙时封顶 48h）

- 状态：原先 4 卡（36928–36931）仅 g1 COMPLETED；g2–g4 TIMEOUT@48h。进度 **50/77**；队列曾空。
- 试加长到 96h/72h/60h → 全部 `QOSMaxWallDurationPerJobLimit`；本集群 **MaxWall=48h**，无法加长。
- 对策：改 `scripts/slurm/submit_paper_baseline_821.sh` 为 **每未完成 fold 一 job** + `--models` 只跑缺 `result.json` 的模型（大模型 + ATCNet 续跑）。
- 已提交 **37966–37973**（fold 3–10，`-t 48:00:00`，PENDING）。已完成：三小 baseline 11/11；ATCNet 8/11；三大模型仅 fold 0–2。

---

## 2026-08-10 — ATCNet arm A/B 全完成；差距分解落地

- arm A job 36481：11/11 ok，`foundation_3c_loso_paper_v1` + atcnet。
  Acc **0.7129±0.1322** / F1 **0.7074±0.1417**（vs 论文 ATCNet 0.6834 / DSGNet 0.6856）。
- arm B job 36504：11/11 ok，`atcnet_3c_loso_paper_recipe_v1`，ALL DONE 19:22，约 20.5h。
  Acc **0.6891±0.1284** / F1 **0.6828±0.1383**。
- Gap 分解（Acc）：模型（flatten−A）**= −5.30pp**；recipe（B−A）**= −2.38pp**（我们的 recipe 更好）；
  论文 ATC − B **= −0.57pp**（B 略高于论文，实现/预处理剩余差基本闭合）。
- 同 recipe 下 5 个 foundation 均低于 ATCNet(A)；最好 flatten 0.6599。难被试 sub-011 两边都 ~0.41–0.42。
- 锚点文档已更新：`4_experiments/wbci_shu/foundation_cross_subject/DSGNET_SHUv5_3C_ANCHOR.md`。
- 下一步（待用户定）：写正式 AI 分析 / 决定是否继续调 foundation / 是否接 DSGNet 复现。

---

## 2026-08-09 夜 — arm B 开启早停 + 修掉「续跑重置 patience」的 bug


- 用户要求 arm B 的 500 epoch 加早停。原设置 `early_stopping_patience: 0` = **没有早停**
  （当初因为论文未提就关掉了）。现改为 **100**，依据：官方 ATCNet 仓库
  `main_TrainValTest.py` L413 的 `train_conf['patience']=100`（其 EarlyStopping 回调被注释掉，
  但这个值是官方写的），且 100/500 与 arm A 的 25/100 是同一 1:5 比例，两条 arm 停机条件可比。
- **`code/training/e2e_trainer.py` bug fix**：续跑时 `patience` 从 0 重新开始，等于每次被抢占
  都白送一整个 patience 窗口，续跑的 run 会比不中断的 run 训得更久。改为从 checkpoint 重建：
  `patience = max(0, start_epoch - best_epoch)`（`best_epoch`/`history` 本来就存在 `last.pt` 里）。
- checkpoint 契约**本来就满足**用户要求：每 cell 只有 `best.pt` + `last.pt`，`last.pt` 带
  optimizer / scheduler / RNG / history，可断点续跑（已实测确认：arm B fold0 的
  `last.pt` epoch=153 / best_epoch=145 / best macro_f1=0.7336）。
- arm B 重新提交 job **36504**（取消 36494），从 fold0 epoch 153 续跑，patience 重建为 8。
- 说明：model selection 仍用 `macro_f1`（与 arm A 一致），**没有**跟随官方的
  `ModelCheckpoint(monitor='val_loss', mode='min')`，这样 A/B 之间只差优化器/batch/epoch/调度，
  归因更干净。官方另有 `ReduceLROnPlateau(factor=0.9, patience=20)`，DSGNet 论文未提，未启用。

---

## 2026-08-10 — 统一对比 run：7 模型 × 8:2:1 × 论文 recipe × 三曲线（已提交四卡）

**目标**：一个 run 里所有模型同协议同配方，且带学长要的 train/val/test 三曲线。
run_id = `paper_baseline_3c_821_v1`，config `code/configs/experiments/paper_baseline_3c_821.yaml`。

**① 三曲线（代码改动）**
- `e2e_trainer.py`：新增 `train_eval_loader` / `test_loader` 两个**只做监控**的 loader，
  每 epoch 评一次写进 `history` 的 `train_eval` / `test`。**test 绝不参与模型选择**：
  `best.pt`、早停、`best_score` 只读 val，守卫写在代码+docstring 里。
  train 曲线取**训练集的固定子集（默认 2000 trial）在 eval 模式**下测，才和 val/test 可比
  （train 模式带 dropout 不可比）；`train_loss` 仍是全量真实训练损失。
- 日志新增紧凑三曲线行：`acc(train/val/test)=0.4065/0.3967/0.3522`。
- 绘图 `scripts/plot_three_curves.py`（per-cell + 跨折均值，均值按最短 fold 截断）。
- 汇总 `scripts/summarize_cross_subject.py` → `tables/{per_cell,per_model,vs_paper}.csv`
  + `REPORT_TABLE.md`（内嵌论文 Table II SHUv5 数字）。

**② baseline 只用官方代码**（`code/models/paper_baselines/`，出处/偏差/排除理由见其 README）
- ✅ EEGNet [18] `vlawhern/arl-eegmodels`（Keras→1:1 移植；逐层核对参数量 3,700）
- ✅ EEGNeX [20] `chenxiachan/EEGNeX`（Keras→1:1 移植；上游无可配结构超参）
- ✅ EEG-Deformer [23] `yi-ding-cs/EEG-Deformer`（**官方就是 PyTorch，原样执行**）
- ✅ ATCNet [24]（8/9 已完成，官方 Keras→移植，参数量 113,732 与官方一致）
- ❌ EEG-Inception [27]：无官方发布（braindecode 那版自己声明未经原作者核对）
- ❌ MDGEEG [35]：占位仓库无代码
- ❌ EEG-DG [38]：**发布代码不完整**——入口 `import Shallow_Inception_Network_2source`
  但仓库无此文件；`DG_Network` 写死 2 个源域（我们有 8 个训练被试）；`Dist_Loss` 全程
  detach 到 numpy 无梯度。改了就不是复现，故按用户规则排除。
- ❌ DSGNet：仅架构预览。其论文数字（0.6856/0.6833）只作引用对照。
- 我们的模型跑 3 个：`dualcd_s4_flatten` / `s4erp` / `dualcd_transformer`。

**③ 协议与配方**
- 划分：8:2:1 跨被试（LOSO，10 个非测试被试里 2 个做验证、8 个训练，每人 3 session；
  trial 数正好 7199:1800:900）。
- 配方：论文 §IV-A（Adam 1e-4 / batch 128 / max 500 epochs / 无 scheduler / 无 wd /
  无梯度裁剪）+ **早停 patience 100**（论文是固定 500，这是记录在案的偏差）。
- **梯度累积（新增）**：两个 67M 模型在 batch 128 下 OOM（smoke 实测）。加
  `train.micro_batch_per_model`（flatten 32×accum4、transformer 16×accum8），
  **优化器仍看到 batch 128**；唯一差异是 BN 统计量落在 micro-batch 上，已在 config 注明。
  `micro_batch_size` 已计入 `cell_signature`。

**④ 验证**：smoke1（7 模型 × fold0 × 3ep）5 通过 + 2 OOM（暴露问题）→ 加累积后
smoke2 两个大模型均通过（flatten 83s/ep、transformer 145s/ep）。三曲线在 5 个 cell 的
`history` 里齐全。

**⑤ Slurm 四卡**（按 fold 切分，每 job 跑全部 7 模型，各 <48h；同命令续跑）：
**36928** folds 0-2、**36929** folds 3-5、**36930** folds 6-8、**36931** folds 9-10。
记录 `outputs/experiments/wbci_shu/paper_baseline_3c_821_v1/parallel_job_ids.txt`。

---

## 2026-08-09 夜 — ATCNet 加第二条 arm：**严格论文 recipe**（把模型差距和配方差距分开）

- 问题：我们 6 个模型用统一 recipe（AdamW+cosine / batch 64 / ≤100 ep / patience 25），
  论文用 Adam 1e-4 / batch 128 / 500 ep。**划分协议一致，配方不一致**，所以不能直接说
  「我们比 DSGNet 差 X pp 是模型差距」。
- 方案（用户选定）：ATCNet 跑两条 arm，**只有 recipe 不同**。
  - arm A `foundation_3c_loso_paper_v1`（models=[atcnet]，job 36481）= 我们的 recipe
    → 与 5 个 foundation 严格同条件可比。
  - arm B `atcnet_3c_loso_paper_recipe_v1`（新 config
    `code/configs/experiments/atcnet_3c_loso_paper_recipe.yaml`）= 论文 recipe：
    Adam lr=1e-4 / batch 128 / 500 epochs / 无 scheduler / 无 weight decay / 无梯度裁剪 /
    **不早停**（论文未提这些，一律关掉而不是继承我们的默认）。
    → 与论文自己的数字可比。Slurm：smoke **36493** → 全量 **36494**（`afterok`）。
- `arm B - arm A` = **recipe 差距**；`论文 0.6834 - arm B` = 剩余的实现/预处理差距。
- 指标口径（用户选定）：只用 **Acc / macro-F1**，不补 Kappa。

---

## 2026-08-09 晚 — ATCNet 改用**官方仓库**实现（作废 braindecode 版）

- 用户要求：必须来自官方 <https://github.com/Altaheri/EEG-ATCNet>。**第一版（braindecode
  PyTorch 移植）已作废**：取消 job 36474、删除 `_atcnet_raw.py` / `_bd_modules.py` 及其
  smoke 产物与 `atcnet__fold0__seed0` cell。55 个 foundation cell 未受影响。
- **官方版落地**（`code/models/atcnet/`）：
  - `_official_keras/{models,attention_models}.py` + `LICENSE` + `UPSTREAM_README.md`
    = 官方仓库**原样**（Keras，不执行，供逐行核对）。
  - `atcnet_torch.py` = 官方 `ATCNet_` 的 **1:1 PyTorch 移植**（Keras 跑不进本项目管线）。
    逐层对齐：`Conv_block_` / `mha_block` / `TCN_block_`、Keras 初始化
    （Conv2D/Dense glorot_uniform、TCN Conv1D he_uniform）、`max_norm(0.6)` 核约束、
    L2 penalty（conv 0.009 / dense 0.5）。
  - `adapter.py` = `{logits, features, confidence}` 契约 + 用 trainer 的
    `uses_custom_loss` 钩子把官方 L2 加回 loss（Keras 是自动折进 loss 的）。
- **保真证据**：用官方 BCI-IV-2a 维度（22ch/1125/4 类）参数量 = **113,732**，
  与官方 README 结果表**完全一致**。我们的维度（58ch/1000/3 类）= 114,719，
  `Tc=17 / Tw=13 / F2=32 / 5 windows`。
- **偏差**（仅 6 条机械性差异，逐条见 `code/models/atcnet/README.md` §3）：数据布局、
  even kernel 的 `same` padding 取整方向、softmax→logits、attention 限定 `mha`、
  L2 由 training_step 加、Keras `MultiHeadAttention`（key_dim 8 ≠ embed 32）需自写。
- **协议**：与 5 个 foundation **完全相同**（同 run_id `foundation_3c_loso_paper_v1`、同
  LOSO session 划分、同 trainer recipe）。**优化器 recipe 与论文不同**（论文 Adam 1e-4 /
  batch 128 / 500 epochs；官方仓库 Adam 1e-3 / batch 64 / 500 epochs；我们 AdamW+cosine /
  batch 64 / ≤100 epochs / patience 25）——写报告必须说明。
- Slurm：smoke **36480**（fold0 / 2 epochs / 隔离输出）→ 全量 **36481**（`afterok`）。

---

## 2026-08-09 — foundation_3c_loso_paper_v1 全量完成（55/55）

- 协议：DSGNet/SHUv5 对齐 LOSO（train ses1–2 / val ses3 / test=留一被试全 session）。
- 5 models × 11 folds × seed0 **全部 ok**；汇总 CSV：
  `outputs/experiments/wbci_shu/foundation_3c_loso_paper_v1/runs/cross_subject__all_models.csv`
- **LOSO mean Acc / F1**（vs 论文 DSGNet 0.6856 / 0.6833）：
  - dualcd_s4_flatten **0.6599 / 0.6569**（最好，−2.6pp Acc）
  - s4erp 0.6477 / 0.6463（−3.8pp）
  - dualcd_transformer 0.6340 / 0.6326（−5.2pp；batch16 才跑通）
  - dualcd_s4_timepatch 0.6107 / 0.6091（−7.5pp）
  - dualcd_s4_pos 0.4511 / 0.4132（远弱）
- 共同难点：sub-011 全模型 ~0.32–0.39；sub-007 最强（flatten 0.868）。
- 结论：**同协议下 5 个 foundation 均未超过论文 DSGNet**；最接近是 flatten。

---

## 2026-08-07 晚 — 续跑 foundation_3c_loso_paper_v1（4 GPU）

- 前一轮 35878–35881 在 ~2h 被 CANCELLED；当时完成 13/55（s4erp 11/11 Acc mean 0.6477；
  timepatch/flatten 各 fold0；pos/transformer 未完成）。
- 重新提交（resume，同 out/ckpt）：**35989** s4erp+pos、**35990** timepatch、
  **35991** flatten(40G)、**35992** transformer(40G)。
- 记录追加：`outputs/experiments/wbci_shu/foundation_3c_loso_paper_v1/parallel_job_ids.txt`

---

## 2026-08-07 — 对齐 DSGNet 论文协议，重跑 WBCIC 3C LOSO（5 foundation）

- **对标论文**：Lou et al., IEEE JBHI 2026，[doi:10.1109/jbhi.2026.3689121](https://doi.org/10.1109/jbhi.2026.3689121)
  （PDF：`inbox/papers/dsgnet_jbhi2026_FullText.pdf` + 根目录同名副本；摘录
  `inbox/papers/dsgnet_jbhi2026_extracted.txt`）。只做其中 **SHUv5 = WBCIC-SHU 三分类 11 人**。
- **论文协议（已落地）**：LOSO；非测试被试 **ses-01+02 → train，ses-03 → val**；测试被试
  **三 session 全测**。实测 fold0：`n_train=5999 / n_val=3000 / n_test=900`（sub-010/ses-01=299）。
- **工程**：新增 `val_mode=sessions`（`cross_subject_protocols.py`）；config
  `foundation_cross_subject_wbci_3c.yaml` → `run_id=foundation_3c_loso_paper_v1`
  （**不复用**旧 `foundation_3c_loso_v1`，旧=留 2 个 val 被试，与论文不一致）。
- **对标数字（论文 Table II，SHUv5，LOSO mean）**：DSGNet Acc **0.6856** / F1 **0.6833** /
  Kappa **0.5284**；次强 ATCNet Acc 0.6834。本轮只跑 5 个 foundation；DSGNet 复现仍 deferred。
- **测试**：session-val 相关 5 passed；dry-run OK。
- **Slurm 4 GPU 并行**（记录：`outputs/experiments/wbci_shu/foundation_3c_loso_paper_v1/parallel_job_ids.txt`）：
  - **35878** `s4erp,dualcd_s4_pos`（32G）
  - **35879** `dualcd_s4_timepatch`（32G）
  - **35880** `dualcd_s4_flatten`（40G）
  - **35881** `dualcd_transformer`（40G）
- 网格：5 models × 11 LOSO folds × seed 0 = 55 cells；同命令可续跑。

---

## 2026-08-05 — 4 GPU 并行拆分 3C LOSO

- 取消单体 job **35359**（已完成 s4erp 11/11 + dualcd_s4_pos 5/11；若继续会抢后面三模型）。
- 新提交（同 out/ckpt，per-model CSV，互不覆盖）:
  - **35433** `dualcd_s4_pos`（续跑剩余 folds）
  - **35434** `dualcd_s4_timepatch`
  - **35435** `dualcd_s4_flatten`（40G）
  - **35436** `dualcd_transformer`（40G）
- 记录：`outputs/experiments/wbci_shu/foundation_3c_loso_v1/parallel_job_ids.txt`

---

## 2026-08-04 — WBCIC 3C LOSO 全量开跑（仅 5 个 foundation；DSGNet 暂不做）


- 用户指示：不管 DSGNet，直接跑五个模型。
- Slurm job **35359**（`e2e_3c_loso`）：`foundation_cross_subject_wbci_3c.yaml`，48h / 32G / gpu2node。
- 修复 `scripts/slurm/shu_gpu.sbatch`：用 `SLURM_SUBMIT_DIR` 找 `_common.sh`（避免 spool 拷贝路径失效）。
- 产出：`outputs/experiments/wbci_shu/foundation_3c_loso_v1/` + `checkpoints/wbci_shu/foundation_3c_loso_v1/`；同命令可续跑。
- 网格：5 models × 11 LOSO folds × seed 0 = 55 cells。

---

## 2026-08-04 — 学长锁定先跑 WBCIC 3C LOSO（11 人）+ 5 foundation + DSGNet


- **协议锁定**：三分类 11 人；LOSO；只存 best+last；断点续跑。二分类 5 折/seed 41–45 **本轮不做**。
- **3C 数据**：`scripts/preprocess_wbci_3c.py` 从官方 `derivatives/3C dataset_processeddata` `.mat`
  转到 `outputs/processed/wbci_shu_3c_mat_clean/`（外部 processed/ 只读）。33 session 全 ok
  （含 sub-010/ses-01 的 299 trial）。标签 {1,2,3}->{0,1,2}。
- **实验 config**：`code/configs/experiments/foundation_cross_subject_wbci_3c.yaml`（dry-run OK）。
- **多类支持**：`normalize_labels` / `evaluate_predictions` 支持 3 类 AUC（macro-OVR）。
- **DSGNet**：GitHub 仅架构预览（`temp_*.py`，*full code upon acceptance*）；源文件已拷到
  `code/models/dsgnet/_*.py`，**尚未**注册进 runner。需向学长确认是否有完整训练代码/超参。
- **下一步**：① 问清 DSGNet 完整代码来源；② 接入 `dsgnet` 到 registry；③ Slurm GPU smoke → 全量 LOSO。

---

## 2026-08-04 — 主线换轨：端到端基础模型 × 跨被试（学长包融合完成，实验未跑）

- **学长新指示**：「先简化任务，先不搞在线学习，先直接搞端到端的模型」——用指定的 **5 个模型**
  （`models_eeg_foundation/` 包：S4ERP + UnifiedDINODualCD_{S4_Pos, S4_Timepatch, S4_Flatten, Transformer}），
  在 WBCIC-SHU 与 SHU 上**分开训练（不合并）**，做**跨被试**实验；要求**只存最后一个 epoch + 最优模型**
  两个 checkpoint，且**支持断点续跑**。
- **Phase 3 → paused（不是废弃）**：`code/tta/`、`4_experiments/*/tta/`、`PHASE3_ROUTE_PLAN.md` 全部保留。
- **融合（done）**：
  1. `code/models/eeg_foundation/`：学长 5 个源文件原样移植（模型数学未改），新增 `adapter.py` 做
     项目契约（`[B,C,T]` 输入 + `{logits, features, confidence}` 输出）与 DualCD 训练钩子
     （`uses_custom_loss` / `training_step` / `after_optimizer_step`）。**4 处偏差**逐条记在该目录 README §4
     （最实质的一条：原包 README 让用户设 `multi_view.low_freq` 但代码里没人读，已改成真参数，
     MI config 用 mu/beta 8–13 / 13–30 Hz）。
  2. `code/training/e2e_trainer.py`：新训练器。只写 `best.pt` + `last.pt`；`last.pt` 带
     optimizer/scheduler/RNG/history；原子写入；`cell_signature` 防「换了划分还复用输出目录」；
     每 epoch 按 `epoch_seed_base+epoch` 重设种子，使续跑与一口气跑完可比。
     **`trainer.py` 一行未改**（Phase 0–2c 可复现）。
  3. `code/experiments/cross_subject_protocols.py`：被试级 `loso` / `kfold_subject` / `holdout`；
     验证集默认取留出的**训练被试**；per-trial z-score（fit-free 无泄漏）；best 与 last 都在每个
     留出被试上评测（CSV 里 `accuracy...` vs `last_accuracy...`）。
  4. runner `foundation_cross_subject` + config `{foundation_cross_subject,shu_foundation_cross_subject}.yaml`
     + 5 个 model YAML + 4 个新 CLI 开关（`--split-protocol/--folds-subset/--monitor/--no-resume`）。
- **验证**：`tests/foundation/` **32 passed（CPU）**，含泄漏守卫、续跑守卫、配置漂移守卫。
  真实 SHU 数据 CPU smoke 跑通「加载→池化→划分→训练→双 checkpoint」。
- **参数量实测**（`n_times=1000`/2 类，@58ch / @32ch）：`s4erp` 1.37M/0.94M、`dualcd_s4_pos` 3.17M/2.32M、
  `dualcd_s4_timepatch` 4.48M/3.63M、`dualcd_s4_flatten` 66.9M/66.0M、`dualcd_transformer` 67.9M/67.0M。
  与学长表格对照：flatten 两个 + Transformer 在 32ch 下吻合 0.4% 以内；`pos`/`timepatch` 实测高 10–16%
  （学长那两行疑似按 ERP 配置 C=21/T=170 量的）。**文档一律以实测为准，不照抄表格。**
- **实测教训**：CPU 上 `s4erp` 单 epoch 734 s（15 s/步，瓶颈是 128 通道 × 1000 点的空间卷积），
  所以**正式 smoke 也必须上 GPU**；已按规矩改走 Slurm（`scripts/slurm/shu_gpu.sbatch`，job 35295 排队中）。
- **内存**：被试数据常驻 ≈6.9 GB（WBCIC）/1.6 GB（SHU）；训练集用 `ConcatDataset` 拼接**不拷贝**
  （若用 `torch.cat` 每个 cell 会再多约 5.5 GB）。WBCIC 全量 Slurm 建议 `--mem=32G`。
- **文献调研（done）**：`inbox/cross_subject_protocol_research.md`。关键事实：两个数据集都**没有**官方
  跨被试划分、两篇数据集论文都**没做**零样本跨被试；WBCIC-SHU 唯一可对标是 **EDAPT（J Neural Eng 2026，
  2-fold 被试划分，零样本 EEGNet 0.81 / DeepConvNet 0.85）**；**SHU 2022 没有任何可验证的已发表跨被试数字**
  且有地板效应（作者跨 session 53.7%，chance 51.4–53.7%，p>0.05）。
- **待办（阻塞点）**：协议参数还没定，`4_experiments/CROSS_SUBJECT_PROTOCOL_MEMO.md` 里 7 个问题
  **需要发给学长确认**（LOSO vs 5-fold、是否严格零样本、session 池化、是否加 EA arm、epoch 预算、
  先跑哪几个模型、SHU 近 chance 怎么写）。**确认前不产出任何被当作结果的数字。**
- **下一步**：等 GPU smoke 出单 epoch 耗时/显存 → 学长拍板协议 → 改 config → 全量（可断点续跑）→
  summarize（summarizer 待写）→ AI 分析报告。

---

## 2026-08-04 — 跨机便携路径改造（portable local configs）

- **目标**：去掉仓库内硬编码绝对路径，方便 GitHub 云端同步、多服务器继续开发。
- **做法**：
  1. 本机路径改为 `*.local.yaml`（gitignore）；仓库保留 `*.example.yaml` + 占位 `paths.yaml` / dataset yaml。
  2. SHU 实验 config 的 `data.manifest` 改为逻辑键 `shu_processed_manifest`；`paths.py` 新增 `resolve_manifest_path`。
  3. Slurm 脚本经 `scripts/slurm/_common.sh` 自动探测项目根；去掉硬编码 `/share/home/yuan/...` 与 mail-user。
  4. `preprocess_shu.py` 默认 out-root 来自 `shu.local.yaml`，不再写死 workspace2。
- **本机**：已生成指向当前共享盘的 `paths.local.yaml` / `shu.local.yaml` / `wbci_shu.local.yaml`。
- **下一步**：需要时 `git add` 便携改动并 push（不要提交 `*.local.yaml`）。

---

## 2026-07-12 — Phase 3 Pretrained-Model Readiness Round（工程验收，非科研裁决）

- **目标**：在学长真实预训练模型交付前，把 TTA 后端从 “embedding replay 已跑通” 提升到
  “mock live inference 已验证 + A0 充分 + 双数据集路由 + 交接规范完备”。**不做**正式 Oracle /
  full T3A / 新算法。
- **多智能体**：Lead + A(审计) + B(live inference) + C(A0/SHU) + D(契约) + E(独立 QA)。
- **Agent A**：当前 smoke 路径无 Critical；Major 含 label-free 仅约定安全、exception taxonomy、
  soft_call、t3a_minimal 近邻 trim≠熵排序、dry_run mkdir 顺序。
- **Agent B**：实现 `ModelInferenceSource`；`AdapterCapabilities` + typed exceptions；
  `run_label_free` 接口级 strip labels；测试专用 mock（Profiles A/B/C）仅在 `tests/tta/support/`。
  测试 **41 passed**（原 14 + 新增）。
- **Agent C**：opt-in `full_a0_replay`（`phase3_tta_full_a0.yaml`）；默认仍 smoke。
  **WBCIC full A0 COMPLETE**：canonical 4320/4320/4320，0 missing/dup，4320 pass，**max_abs_delta=0.0**。
  **SHU smoke passed**：2 cells no_tta，路径均 `outputs/experiments/shu/`，max\|Δ\|≈2.1e-7。
  修复 dry_run 无 I/O；SHU `run_t3a/run_oracle=false`。
- **Agent D**：权威契约 `3_online_adaptation/PRETRAINED_MODEL_INTEGRATION_CONTRACT.md`（Lead 已同步状态表）。
- **Agent E**：独立复核 1–9/11–12 PASS；契约曾 stale（已修）。结论工程上 **Ready to receive real pretrained model**。
- **未做**：学长模型接入、formal Oracle、full T3A、Tent/SHOT、改写 Phase 2c。
- **下一步**：学长交付 checkpoint → 新 adapter → preflight → live smoke → 再 Phase 3B。

## 2026-07-10 — Phase 3 Round-1：model-agnostic TTA backend scaffold + smoke runnable

- **目标**：搭模型无关 T3A/TTA 后端骨架 + 最小测试/smoke；**不是** full T3A experiment，不写科研结论。
- **新增包 `code/tta/`**：adapters（Protocol + registry + embedding_only + baseline_torch 示例）/
  feature_sources（FeatureBundle + embedding_replay 路径重拼 + model_inference scaffold）/
  methods（no_tta + t3a_minimal）/ oracle（label_guard + target_label_oracle_proto）/
  eval（schema/metrics）/ report（smoke reporters）+ `method_catalog.yaml`。
- **入口**：`code/experiments/session_tta.py`；`code/runners.py` 已注册 `phase3_tta`；
  config `round1` 安全开关（默认 smoke / max_cells=4）；`code/methods/t3a.py` 薄 re-export。
- **测试**：`tests/tta/` 14 passed（CPU，`mi_torch`）。
- **WBCIC smoke（CPU）**：自动选 sub-005(stable)+sub-020(high)，seed0，eegnet 示例 adapter；
  3 cells × (no_tta + t3a_minimal + oracle) = 9 rows。
  - **A0 replay**：3/3 cell `|Δ|=0` vs Phase 2c `acc_target`（路径重拼成功，未改 Phase 2c 产物）。
  - **A1**：pipeline runnable；smoke 数字**不得**解读为 T3A 有效/无效。
  - **Oracle**：minimal target-label proto 跑通，`used_target_labels=True` 等三字段齐全；阈值仍 provisional。
- **结果区**：`4_experiments/wbci_shu/tta/{smoke,replay_validation,oracle_diagnostic,method_catalog,reports}/`；
  heavy `outputs/experiments/wbci_shu/tta_v1/`。SHU 侧 catalog+README 已建，Round-1 未跑 SHU smoke。
- **未做**：full sweep / ablation / Tent/SHOT / catalog 复杂方法实现 / 预训练模型接入 / 全量 A0。
- **下一步**：等学长预训练模型 → 新增 adapter+config；可选扩大 A0 全量 replay；再按 Oracle 裁决是否扩大 T3A。

## 2026-07-08（晚）— 学长批准 Phase 0 + A0/A1，补 7 条硬约束（v2.1，未跑代码）

- **学长审批**：Phase 3 v2 主路线通过（Oracle 提前为 Phase B 裁决门不推翻）；批准开 **Phase 0 + Phase A0/A1**；
  附 **7 条上机前硬约束**，其中 cell_id / Oracle 泄漏标识 / Mahalanobis 数值稳定性 三条"不补后面结果无法 defend"。
- **7 条硬约束（已写进 `PHASE3_ROUTE_PLAN.md` §2.5 + 对应 Phase）**：
  1. 结果目录**定死 `4_experiments/{wbci_shu,shu}/tta/`**；`3_online_adaptation/` 只放设计文档。
     → 已改两个 config 的 `output.readable_dir`（原 `3_online_adaptation/*/tta` → `4_experiments/*/tta`）+ DESIGN 文件清单。
  2. A0 定义 **canonical cell_id**（dataset/model/seed/subject/source/target/cell_id/npz_path_resolved/n_target_trials），
     No-TTA 复现按 cell_id join（先断言两侧 cell_id 集合一致），防错行 join。
  3. A1 stable/high-drift 被试**从 Phase 2c drift tertile 表自动选**，落盘 `selected_smoke_cells.csv`，禁手选（防 cherry-pick）。
  4. Oracle 方法**带 `used_target_labels` 泄漏标识**，报告单列 "Oracle diagnostic only"，不与 deployable T3A 混主表。
  5. Mahalanobis/shrinkage 默认 **shrinkage cov + ridge eps + 记录 cond number**，病态则 fail 或降级 cosine oracle（标 degraded）。
  6. Phase B 裁决门 = **收益+风险双条件**：均值>+3pp 且 high-drift/各模型无高负迁移且 Fisher 有恢复 → 才进 full；
     <+1pp → 转 scatter/reliability/decision-boundary；割裂情形降级小范围验证。
  7. **FBCNet 与 SHU 单列**，不与 WBCIC EEGNet/DeepConvNet 混成主结论（主结论=WBCIC×{EEGNet,DeepConvNet}）。
- **本轮改动（纯文档/配置，未跑代码）**：`PHASE3_ROUTE_PLAN.md` 升级为 v2.1 批准版（新增 §2.5 七条 + Phase A0/A1/B/E
  逐条落约束 + §4 决策锁定 + §6 审批记录）；两个 `phase3_tta.yaml` 的 readable_dir 改到 4_experiments；
  同步 `PHASE3_TTA_DESIGN.md`（文件清单路径）、`AGENTS.md`、`0_docs/STATUS.md`。另交付 `PROJECT_ARCH_SYNC_FOR_ADVISOR.md`
  （给学长的架构同步简报，防结构冲突）。
- **下一步（已获批，可开工写代码）**：Phase A0 = `code/experiments/session_tta.py` + 注册 `phase3_tta` runner +
  No-TTA 按 cell_id 精确复现（|Δ|<1e-6），逐条对照 7 条硬约束；通过后 Phase A1 minimal T3A smoke。

## 2026-07-08 — Phase 3 路线 v2 重排：Oracle 提前为裁决门 + Phase 0 状态修正（planned，未跑代码）

- **触发**：学长用 `writing-plans` 口径给出修订路线，核心改动 = **Oracle 上限实验必须提前到 T3A 大扫之前
  作为科研裁决门**（原 v1 把 Oracle 放最后 Step E）。依据 = Phase 2c 权威结论：prototype drift 只解释部分
  掉点（多元 R²≈0.35），主机制是 within-class scatter 膨胀 / Fisher collapse（**非 centroid collapse**）。
  若连"已知 target 原型/scatter-aware 几何"的理论上限都救不回多少 acc，T3A（带噪伪标签重估原型）更不可能有大收益。
- **架构映射**：学长文档按另一套项目结构写（P10 / CLAUDE.md / `run_phase()` / 无 runners.py/summaries）。
  已全部映射到本项目真实结构：灵魂记忆是 `AGENTS.md`（`CLAUDE.md` 仅指针）；runner 走 `code/runners.py` 的
  `PHASE_RUNNERS`；已有 `code/summaries/`；Phase 2c 表在 `4_experiments/{wbci_shu,shu}/prototype_drift/tables/`
  （`trial_embeddings_index.csv` / `prototype_drift_metrics.csv` 均在）；结果放 `4_experiments/{wbci_shu,shu}/tta/`。
- **本轮产出（纯文档，未写代码、未跑实验）**：
  1. 新建根目录 `PHASE3_ROUTE_PLAN.md`（完整 v2 路线 Phase 0→G + 决策门 + 待定选项 + 批准口径，供用户过目审批）。
  2. 重排 `3_online_adaptation/PHASE3_TTA_DESIGN.md` §7 实现路线 + 时间线：Oracle 从 Step E 提前为 **Phase B 裁决门**。
  3. Phase 0 状态修正：`AGENTS.md`（当前主线/第一优先/逻辑链/事实 → Oracle 先裁决）、`0_docs/STATUS.md`、
     `README.md` §0、`results.md`（WBCIC Prototype Drift 行措辞 Step 4→Phase 3B Oracle）。
- **裁决门槛**：Oracle >+3pp → 继续 T3A ablation/safe-T3A/全量；<+1pp → 停大规模 T3A，转 scatter/reliability/
  decision-boundary 机制；中间地带谨慎推进。三种结果对应论文三条叙事路线（见 PHASE3_ROUTE_PLAN §3 Phase G）。
- **🐞 已定位 bug（实现时消费端修，不改 Phase 2c 产物）**：WBCIC `outputs/experiments/wbci_shu/prototype_drift_v1/
  runs/embed_index__*.csv` 的 `npz_path` 列是失效旧路径（写成 `outputs/experiments/prototype_drift_v1/embeddings/...`，
  缺 `wbci_shu/`，文件已迁移）；SHU 的 index 正确。Phase 3 runner 须用 config `source_embeddings.embeddings_dir`
  重拼路径、缺失即 fail-fast，不信任该列。已确认 `phase3_tta.yaml` 里 embeddings_dir 指向正确新路径。
- **纪律**：T3A 现仅作无梯度 prototype baseline，Oracle 未裁决前不写"最终修复方案"、不做全量；未跑不写 done。
- **下一步（等用户一句话确认再写代码）**：Phase A0 = `code/experiments/session_tta.py` + 注册 `phase3_tta` runner
  + **No-TTA 逐 cell 精确复现 Phase 2c `acc_target`（|Δ|<1e-6）**；通过后 Phase A1 minimal T3A smoke（WBCIC/EEGNet/
  2 被试/seed0/src_proto+cosine）；再 Phase B Oracle 裁决。

## 2026-07-06 — Phase 3 立项：基于原型的测试时适应 (T3A) 方案 + 路线文件（planned，未跑）

- **触发**：用户给出论文 `2604.16926v1`（Lee, Pradeepkumar, Sun. *Test-Time Adaptation for EEG
  Foundation Models: A Systematic Study under Real-World Distribution Shifts*，NeuroAdapt-Bench），
  要求基于 T3A 做跨 session 脑电模型优化：用预训练模型抽特征、按不确定性（熵）筛高置信样本作伪标签
  微调、提升泛化；1 个月内组合"不同预训练模型 × 不确定性指标"找最优组合并对比 baseline。本条只**立项 +
  写方案 + 更新指示文件**，未实现 runner、未跑实验、未产出任何结果数字。
- **读论文（HTML 全文，非编造）**：本机无 PDF 库且不装包，改用 arXiv HTML 全文
  (`arxiv.org/html/2604.16926v1`) 提取正文。要点：① 系统 benchmark 三种 TTA——**Tent**(梯度，改 norm
  affine)、**SHOT**(梯度，改特征提取器 + 互信息 + 伪标签)、**T3A**(无优化，原型 + 熵筛选)。② 核心结论：
  **只有 T3A 在 in-dist/OOD/极端漂移三设定下平均平衡准确率为正、且最稳**；梯度类常负迁移；T3A 对 batch
  size 不敏感、类不平衡时收益最大（REVE-Base@CHB-MIT +18.9pp）。③ T3A 超参 `filter_k=20`、episodic=False、
  预测 `p(y=k|z)∝exp(z·c_k)`；c_k=每类低熵 support 特征均值。④ 论文用**EEG 基础模型**(CBraMod/REVE/
  TFM-Tokenizer)冻结编码器 + 共享线性头，跑**跨数据集/任务/模态**漂移。
- **映射到本项目（关键契合）**：Phase 2c 已证跨 session 掉点 = within-class scatter 膨胀/Fisher collapse
  (非 centroid collapse)、prototype 确漂移、cosine>euclidean。**T3A 正是"无标签用 target 特征重估 class
  prototype 调整分类器几何"的可部署解**，也是原 Step 4 cosine Oracle 上限(label-informed)的 label-free 对应物。
  故把原 Step 4 并入 Phase 3，统一为「基于原型的测试时适应」：Oracle 上限(诊断) + T3A/Tent/SHOT(方法)。
- **代码复用盘点（已读源码确认）**：三模型(eegnet/deepconvnet/fbcnet)均 `.classifier=nn.Linear` +
  `{logits,features,confidence}` 契约 → T3A 可用分类头权重 ω_k 初始化 support。`prototype_drift.py` 已
  存每 (subject,direction,model,seed) 的 `target_test__{z,logits,probs,pred,conf,y}` npz + checkpoint →
  **T3A 无优化、纯特征空间，全程离线 CPU replay 即可，无需重训/GPU**。runner 模式(`PHASE_RUNNERS`)、
  `--summarize` 调度、config 结构均已摸清。
- **诚实要点（写进 DESIGN，避免邀功/误导）**：① backbone 是自训小模型≠论文基础模型，跨 session≠跨数据集，
  不得声称基础模型结果；② **二分类预测熵是最大 softmax 概率的严格单调函数 → "熵筛选"与"最大置信度筛选"排序
  完全等价**，本任务真正有区分度的轴是 filter_k / 几何(cosine vs dot) / 软硬(soft 加权 vs hard top-k) /
  margin，报告须点明；③ SHU 近 chance，伪标签噪声大，预期收益有限；④ FBCNet 几何异常，T3A 恐失效，单列。
- **本次产出（planned，非结果）**：新增路线文件 `3_online_adaptation/PHASE3_TTA_DESIGN.md`（方法/协议/
  1 个月矩阵/实现 Step A–F/时间线/诚实警示）；config 骨架 `code/configs/experiments/{phase3_tta,
  shu_phase3_tta}.yaml`（runner 待实现，现在跑会报 "No runner registered"，与 future_*.yaml 同惯例）；
  同步 AGENTS.md（主线/逻辑链/事实）、STATUS.md、FILE_CATALOG.md、operation_log.md。
- **下一步（Step A，照 DESIGN §7）**：实现 `code/methods/t3a.py`（support/原型/熵筛选/cosine|dot，无梯度）
  + `code/experiments/session_tta.py`（离线 replay Phase 2c 嵌入，No-TTA + T3A，泄漏断言 + NaN fail-fast）
  + 注册 `phase3_tta` runner + 接入 `--summarize`；CPU smoke（subjects 1,2，No-TTA acc 必须与 Phase 2c
  `acc_target` 对齐）。之后 Step B Tent/SHOT(GPU)、Step C 汇总、Step D 全量 sweep、Step E Oracle、Step F 分析。

## 2026-07-06 — SHU Phase 1/2a/2b/2c summarize + AI 分析补齐（与 WBCIC 齐平）

- **目标**：把 SHU 剩余进度（训练早已完成、仅缺 summarize+AI 分析）一次补齐，使 SHU 在 Phase 0–2c 与 WBCIC 齐平。全程只读磁盘真实产物，数字均来自各阶段 `tables/`，未捏造。
- **执行顺序**：P1 → P2a → P2b → P2c，均 `python code/run.py --summarize --config <shu_phaseX>`（CPU，`mi_torch`）。
- **两处 bug 修复（否则 SHU 结果不对）**：
  1. **P2c summarizer manifest key**：`code/experiments/prototype_drift_summarize.py:summarize_from_cfg` 原读 `data.manifest_path`，但全项目约定是 `data.manifest`（见 `runners._resolve_manifest`）。SHU 因此回退到 WBCIC manifest，`build_run_status` 误用 WBCIC 期望网格（把 sub-026+ 判 missing，假报 2220 missing）。改为优先读 `data.manifest`。修正后 SHU 期望网格正确，**run_status 7500/7500 全 ok**。
  2. **P2b baseline schema**：Phase 2b 的 `none_reference` join 需要 `acc/bacc/f1 + train_sessions + training_scope`，而 Phase 1 cross 产出是 `accuracy/balanced_accuracy/macro_f1 + train_session`。新增数据集无关脚本 `scripts/make_baseline_cross_all.py` 做 schema 适配，写到 config `baseline_cross_all` 路径（7500 行，training_scope=single_source）。join key (model,train_sessions,test_session,subject,seed) 行对行匹配。
- **完整性核验**：P1 within 18750 + cross 7500 行（3 models×5 seeds 全齐）；P2a 375 行 ok；P2b 45000 行 ok / 0 failed / 0 NaN / complete=True；P2c 45000 metric 行、run_status 7500 全 ok。
- **SHU 真实结果**（数字见各 `tables/` 与 AI_ANALYSIS）：
  - **P1 baseline**：within/cross（5-seed）EEGNet 0.611/0.538、DeepConvNet 0.606/0.536、FBCNet 0.553/0.508；drop 7.3/7.0/4.5pp。**cross 近 chance（地板效应）**，掉点 pp 小≠更稳。排序同 WBCIC，FBCNet 最弱。
  - **P2a multi-source**：ses-01+02→03 = EEGNet 0.544 / DeepConvNet 0.558 / FBCNet 0.512；vs 最强单源 +0.7 / +2.6 / −0.2pp（方向同 WBCIC）。
  - **P2b alignment**：无对齐 0.5274；最佳 **session_zscore +1.42pp**（≠WBCIC 的 BN-stats），4/5 净正，filterbank −1.47pp 有害；**无方法过 +2pp**；high-drift 受益最小（z-score stable +2.88/moderate +1.06/high +0.44pp）。
  - **P2c prototype drift**：机制同 WBCIC——within-class scatter 膨胀 **15.7→38.3 (+144%)** / Fisher collapse **1.96→0.79 (−60%)**，非 centroid collapse（separation 6.6→23.9 反增）。最强预测子 fisher_change ρ=0.43、separation_change cosine ρ=0.38/r²=0.16；drift_mean/direction/margin 弱。cosine>euclidean。EEGNet/DeepConvNet 清晰、**FBCNet 弱且几何异常**（scatter 几乎不变、separation 反缩）。信号比 WBCIC 更噪（cross 近 chance，acc_drop 三分位非单调）。
- **可读产物落地**：P1+P2a → `2_baseline/shu/no_alignment_baseline/{tables,figures,report}`（并入同一目录，未新建 phase 子目录）；P2b → `2_baseline/shu/alignment_baseline/`；P2c → `4_experiments/shu/prototype_drift/`。每个 report/ 都写了 `AI_ANALYSIS.md`（9 段，数字源自 tables）。
- **诚实提醒**：P1 脚本原生报告内嵌 WBCIC 参照常量（论文 85.32/148 sessions/288 pairs/ses01-03 trend），对 SHU 不适用、并因此假报 "INCOMPLETE"；已在 AI_ANALYSIS §0 明确标注，以 tables 真实数字为准。
- 状态：**SHU Phase 0–2c 全部 done**，与 WBCIC 齐平。下一步 = Step 4 cosine Oracle 上限诊断（qualified go），FBCNet 单独处理。

## 2026-07-05 — 重新接手：盘面核查（SHU 训练全完成，summarize/AI 分析仍缺）

- **背景**：用户重新接手项目，并准备用 ChatGPT 当"指挥"来驱动 Cursor agent。本条只做**盘面事实核查**，未跑新实验、未 summarize、未改结果数值。
- **核查方法**：`squeue`（空，无在跑/排队 job）+ 逐 phase 数 run CSV + 扫 `logs/slurm/shu_*.out` 是否有 Traceback/CUDA/RuntimeError。
- **SHU 训练：4 个 phase 全部训练完成，无失败**（最后一个 CSV 落盘 2026-06-13 05:23，此后无活动）：
  - Phase 1 baseline：`session_model_compare_v1/runs/` 30 CSV = within/cross/meta × 3 models × seeds 0-4；日志 `ALL DONE`。
  - Phase 2a multisource：`session_multisource_v1/runs/` 15 CSV（3 models × seeds 0-4），ses01+02→03。
  - Phase 2b alignment：`alignment_baseline_v1/` 75 CSV（5 methods × 3 models × seeds 0-4）；日志 `rows=2500 ok=2500 failed=0`。
  - Phase 2c prototype drift：`prototype_drift_v1/runs/` 60 CSV + 15 `metrics__*.csv`（3 models × seeds 0-4）。
- **关键 gap（下一步动作）**：**summarize + AI 分析尚未跑**。可读结果区 `2_baseline/shu/`、`4_experiments/shu/prototype_drift/` 除 README 外**为空**；无聚合 `*_metrics.csv` / canonical `REPORT.md` / `run_status.csv`。SHU 的 P1/2a/2b/2c **正式准确率/漂移数字尚不存在**（不得引用/编造）。
- **唯一已落地的 SHU 可读结果**：Phase 0 漂移诊断（`1_session_drift/shu/` + `report/AI_ANALYSIS.md`）。
- **诚实状态**：SHU Phase 1/2a/2b/2c = **训练 done，summarize+AI 分析 pending**。这是重新接手后的第一优先动作。
- **下一步（按序）**：① `python code/run.py --summarize --config code/configs/experiments/shu_phase1_baseline.yaml`（P1 single-source cross）→ 归并进 `2_baseline/shu/no_alignment_baseline/`；② P2a multisource summarize（并入同一 `no_alignment_baseline/`，见 2026-06-12 13:27 归属约束）；③ P2b alignment summarize → `2_baseline/shu/alignment_baseline/`；④ P2c summarize → `4_experiments/shu/prototype_drift/`。每步核验 run_status 后再写 AI 分析，并同步 5 个 handoff 文件。之后才是 WBCIC 主线的 Step 4（cosine 空间 Oracle 上限诊断）。

## 2026-06-11 — SHU 主线复跑启动：Slurm 脚本 + Phase 0 漂移诊断完成

- **目标**：把 WBCIC 已完成的跨 session 主线（Phase 0/1/2a/2b/2c）在 SHU 2022 上按双数据集并列架构原样复跑。先不碰 online/adaptation/agent/41-10。
- **Step 1（只检查）**：4 个 SHU config dry-run 全 OK，输出全部 `shu/` 作用域，无 WBCIC 覆盖；manifest 125 ok / 25 subj；`outputs/experiments/shu/` 空、`outputs/analysis/shu` 与 `checkpoints/shu` 不存在；env `mi_torch_cu118` 在；分区仅 gpu2node/gpu3node。
- **Step 2（Slurm）**：新增 `scripts/slurm/shu_gpu.sbatch`（GPU 训练）+ `shu_cpu.sbatch`（CPU 漂移/汇总），均用 `mi_torch_cu118`、cuda fail-fast、GPU 不在登录节点跑。WBCIC 的 prototype_drift sbatch 是写死的，故新建 SHU 通用脚本。
- **发现的 gap**：① SHU 缺 Phase 2a multisource config（待 step 5 给最小补充方案）；② Phase 0 的 `per_subject_drift_summary.csv`/`session_pair_summary.csv` 不是新 runner 产物，legacy builder 硬编码 3 session → 新建 session 无关的 `scripts/build_drift_report.py`。
- **Step 3（Phase 0 完成）**：Slurm CPU job 21601，250 pairs / 25 subjects / 367s。
  - 核心：MMD 0.356、CSP_sim 0.344、ERD μ/β 0.527/0.532、RMS median 1.03、fisher_shift≈-0.0012。
  - 机制与 WBCIC 同质（空间+频谱漂移，幅值稳定，可分性不塌），但 SHU 空间漂移**更重**（MMD 更大 0.356>0.238、CSP 更低 0.344<0.420）。
  - 分层 high 9 / moderate 8 / stable 8；最漂移 sub-017/008/019，最稳定 sub-002/006/020。
  - pair 非单调：01-04 MMD 最大(0.413)、02-03 最小(0.284)。
  - 结果落 `1_session_drift/shu/{report,tables,figures}/`（含 14 图）+ `report/AI_ANALYSIS.md`（9 段，数字源自 tables）。
- **Step 4 Phase 1**：3 个 GPU job（每 model 一个，job 21602-21604，gpu2node），within 10-fold + directed cross，seed 0。running。
- **并行提交（2026-06-11 22:28）**：把后续所有可并行训练任务一次性提交（训练彼此独立，只有 summarize 有依赖）。
  跨 gpu2node/gpu3node 交替分配。job ids 记于 `outputs/experiments/shu/_job_ids/shu_full_2228.txt`：
  - **Phase 2a** multisource（ses-01+02→ses-03）：21610-21612（每 model 一个，含 5 seeds）。
  - **Phase 2b** alignment：21613-21627（每 model×seed 一个，5 methods × single pairs）。
  - **Phase 2c** prototype drift：21628-21642（每 model×seed 一个，20 有向对）。
  - 已 smoke 确认 phase2a eegnet 正常启动（25 eligible / 0 skipped）。共 36 jobs（4 R / 32 PD）。
- **待训练完成后（下一轮）**：① phase1 summarize → `2_baseline/shu/no_alignment_baseline/` + AI 分析；
  ② phase2a summarize（base 路径已 config 化）；③ 解决 phase2b `baseline_cross_all` 列schema 适配后 summarize → `alignment_baseline/`；
  ④ phase2c summarize → `4_experiments/shu/prototype_drift/`。每步写 AI 分析并同步 5 个 handoff 文件。
- 状态：Phase 0 **done**；Phase 1/2a/2b/2c 训练 **submitted/running**，summarize+AI 分析 pending。

## 2026-06-12 13:27 — 目录归属约束（summarize 前必读）

- **Phase 1 baseline 与 Phase 2a multi-source 不是独立结果区**，是 no-alignment baseline 下的两个协议/子结果。
- SHU baseline 最终报告/汇总表/图必须**统一归并**到 `2_baseline/shu/no_alignment_baseline/{report,tables,figures}/`
  （与 WBCIC 的 `2_baseline/wbci_shu/no_alignment_baseline/` 结构一致：within + single-source cross + multi-source 合在一起）。
- **禁止新建** `2_baseline/shu/phase1_baseline/` 或 `2_baseline/shu/phase2a_multisource/`。
- 训练/汇总可分步：先 summarize single-source baseline（P1），再做 multi-source comparison（P2a，依赖 P1 的 cross 表），
  但两者产物都落进 `no_alignment_baseline/`。Phase 2a 的角色是与 Phase 1 single-source cross baseline 对比。
- 训练阶段不受影响（outputs/checkpoints 仍按 run_id 分 session_model_compare_v1 / session_multisource_v1）；
  归并只发生在可读结果区 `2_baseline/shu/no_alignment_baseline/`。

## 2026-06-11 23:15 — Seed 覆盖核查（无覆盖/无重复风险确认）

只核查，未 summarize。结论：seed 配置正确，无重复跑 0-4 风险。
- **Job 状态**：48 jobs。Phase 2a（21610-12）COMPLETED；21602-4 + 21613 RUNNING；其余 PENDING。无 FAILED（ExitCode 全 0:0）。
- **21602-4（旧 seed0）**：日志明确 `seeds=[0]`（22:07 解析，早于 22:44 改 config）→ 仅 seed0。config 是进程启动时一次性读取，后改不影响在跑 job。
- **21644-55（补交）**：`scontrol`/`sacct SubmitLine` 确认命令显式 `--seeds N`（CLI 覆盖 config），单 seed 限定 → **不会重复跑 0-4**。
- **Phase 1 输出**：runs 目录仍空（seed0 job 还在 within 阶段，sub-009/25，CSV 在每 protocol×model 完成后才落盘）。提交层面 seed0(21602-4)+seed1-4(21644-55)=0-4，per-seed 文件名互不覆盖。
- **Phase 2a**：COMPLETED，3 models × seeds 0-4，每 CSV 单 seed、25 行（25 subj，ses01+02→03），无重复。
- **Phase 2b**：21613 running 日志 `seeds=[0]`、5 methods、tasks=500；21613-27 per model×seed → 0-4。
- **Phase 2c**：21628-42 per model×seed（pending）→ 0-4。
- **job id 记录**：补交 21644-55 原记于单独文件 `shu_phase1_seeds1-4_2244.txt`；本轮在 `shu_full_2228.txt` 末尾**追加**引用（未改历史行）。
- **下一步**：仅等待训练跑完（受 QOS 4gpu/分区限速，分批）。无需补交/取消。完成后按 P1→P2a→P2b→P2c 做完整性核查，再 summarize。

## 2026-06-11 22:44 — 修正 Phase 1 seed 覆盖（对齐 WBCIC 5-seed 标准）

- **发现的 bug**：Phase 1 baseline 首批提交（21602-4）用 `--models X` 未带 `--seeds`，而 `shu_phase1_baseline.yaml`
  原 `train.seeds: [0]` → **只跑了 seed 0**，不符合 WBCIC 标准主线（5 seeds）。
- **修正**：① 把 `shu_phase1_baseline.yaml` 的 `train.seeds` 改为 `[0,1,2,3,4]`（标准化）；
  ② 补提交 Phase 1 seeds 1-4，每 (model,seed) 一个 job（21644-21655，共 12 个），跨 gpu2node/gpu3node 交替。
  per-seed CSV 命名（`{tag}__seed{sd}.csv`）天然不覆盖已在跑的 seed0，无需新 run_id。
  id 记于 `outputs/experiments/shu/_job_ids/shu_phase1_seeds1-4_2244.txt`。
- **seed 覆盖核查（提交层面）**：
  - Phase 1：seed0=21602-4(running) + seed1-4=21644-55 → **0-4 覆盖**。
  - Phase 2a：21610-12，config 内含 seeds [0-4] → **0-4 覆盖**。
  - Phase 2b：21613-27（每 model×seed）→ **0-4 覆盖**。
  - Phase 2c：21628-42（每 model×seed）→ **0-4 覆盖**。
- **集群约束**：QOS=32cpu4gpu64g/partition，每 job -c8/gpu1 → 每分区最多 4 GPU 并发（双分区共 8）；其余排队（QOSMaxCpuPerUserLimit）。48 jobs 会分批跑完。
- **纪律**：训练完成前**不执行 summarize、不写 AI_ANALYSIS、不更新 results.md 正式数值**。完成后顺序 summarize：P1→P2a→P2b→P2c，每步核验再写 AI 分析。

## 2026-06-11 — SHU 接入 + 全项目改双数据集并列架构

- **目录架构**：1/2/3/4/5 各结果区全部改为 `wbci_shu/` 与 `shu/` 并列；2_baseline 下保留
  `no_alignment_baseline/` + `alignment_baseline/`，再到 `report/tables/figures/`。每一层目录都有 README
  （生成器 `scripts/scaffold_readmes.py`）。现有 WBCIC 结果迁入 `wbci_shu/`，SHU 侧建空骨架。
- **outputs/checkpoints 也并列**：`outputs/experiments/{wbci_shu,shu}/<run_id>/`、
  `outputs/analysis/{wbci_shu,shu}/`、`checkpoints/{wbci_shu,shu}/<run_id>/`。5 个 WBCIC config 输出路径
  全部加 `wbci_shu/` 前缀；已物理迁移现有 prototype_drift 工件。
- **SHU 预处理完成**：入口=作者发布 per-session `.mat`（已带通/陷波/4s 切段/250Hz/32ch），仅做标签
  {1,2}->{0,1} 归一化转存 `.npz`，不二次预处理。脚本 `scripts/preprocess_shu.py` +
  `code/preprocessing/shu_mat.py`。输出 `/share/workspace2/moto_imagination/SHU/processed/npz_clean/`，
  **125 session 全 ok / 25 subjects**，manifest 列与 WBCIC 一致。
- **命名修正**：原 `mat_clean` 目录改名 `npz_clean`（实际存的是 npz，避免误导），manifest 路径与 config 同步。
- **runner 数据集解耦**：`code/runners.py` 新增 `_resolve_manifest(cfg,P)`，manifest 由 config `data.manifest`
  决定，WBCIC 缺省回退 paths.yaml。新增 4 个 SHU config `shu_phase{0,1,2b,2c}_*.yaml`（同一批 runner）。
- **checkpoint 命名自文档化**：新增 `checkpoints/README.md`，说明任务前缀 within/cross/multisource/
  single/multi/proto + dataset/run_id/method/model/sub/session/seed。
- **验证**：9 个 config 全部 dry-run OK；SHU phase1（within 10 + cross 20 对）、phase2b（EA/RA 32ch 40 行全 ok）
  CPU smoke 端到端通过。
- **边界规则更新**：workspace2 仅 `processed/` 子树可写，raw 只读（AGENTS/STATUS/README 同步）。
- 状态：SHU **data ready，实验 pending**（config 就绪，未跑全量）。下一步可直接对 SHU 开跑 Phase 0/1/2b/2c。

---

## 2026-06-11 — Phase 2c 全量完成 + AI 深度分析报告

- Full Slurm 全部 COMPLETED（16 jobs exit 0，GPU 各约 1h，summarizer 15s）。验证：run_status.csv
  4320 cells 全 ok；metrics 25920 行（25790 ok + 130 correct_only degenerate_empty_class）；
  used_target_labels_for_training 全 False、n_target_labels_used_for_training≡0；ok 行无 NaN/Inf。
- 关键结果（canonical label_based/euclidean, n=4320）：所有几何信号显著但中等。
  separation_change ρ=0.389、fisher_change 0.359、drift_mean 0.352、neg_margin 0.313、
  direction_cosine −0.237。多元标准化线性模型 R²≈0.35。
- 机制结论：不是 centroid collapse，而是 within-class scatter 膨胀。源→目标：sep 8.45→11.78（更远）、
  scatter 16.86→25.83(+53%)、Fisher 4.58→1.57(−66%)。即表征"弥散/糊化"。
- 方法学发现：cosine 几何比 euclidean 更线性（separation_change cosine R²=0.176 vs euclidean 0.001）；
  下游 prototype 方法应在 cosine/归一化空间做。
- 漂移分层：low/mid/high drift tertile 的 acc_drop = 0.048 / 0.141 / 0.155 → 掉点集中在高漂移 cell。
- 模型依赖：EEGNet/DeepConvNet 漂移信号清晰（drift ρ 0.56/0.49），FBCNet 很弱（direction_cosine≈0，
  neg_margin 0.095）→ FBCNet 掉点是另一种几何成因，不能并入 prototype 结论。
- 方向不对称复现：ses-03 作 source 最难（ses-03→01 0.150）；ses-02→03 最易（0.086）。
- 写出 `4_experiments/prototype_drift/report/AI_ANALYSIS.md`（9 段，全部数字来自 tables，未捏造）。
- 分叉判定：**qualified go**。Step 4 先做 cosine 空间 Oracle 上限诊断 + scatter/reliability 探针，
  不无条件承诺 prototype adaptation；FBCNet 单独处理；drift 用作 online trigger。
- 状态：Phase 2c **done/complete**。

## 2026-06-11 — Phase 2c Prototype Drift Analysis 实现 + 全量 Slurm 提交（submitted/pending）

- 新增 `code/configs/experiments/phase2c_prototype_drift.yaml`：3 models (eegnet/deepconvnet/fbcnet)、
  5 seeds (0-4)、6 directed pairs、prototype_types=[label_based, confidence_weighted, correct_only]、
  distances=[euclidean, cosine]；source-only 训练、target test-only、target_label_usage=offline_diagnostic_only。
- 新增 `code/experiments/prototype_drift.py`：复用 session_splits/trainer/registry；对每个 subject×source×
  target×model×seed 在 source 上训练（source-train 切 val 做 early stopping），冻结后提取 source_train/
  source_val/target_test 的 penultimate embedding + logits/probs/pred/conf（fallback conf=max softmax），
  计算 label/confidence/correct prototypes，输出 6 类漂移指标（drift / direction_cosine / separation /
  margin(neg rate) / scatter / fisher）。加泄漏断言（source≠target、train/val 不交叠、
  n_target_labels_used_for_training≡0）+ NaN/Inf fail-fast；degenerate（correct_only 空类）显式标 status，不写坏数。
- 新增 `code/experiments/prototype_drift_summarize.py`：合并 per-(model,seed) CSV → 
  prototype_drift_metrics.csv / prototype_table.csv / prototype_accuracy_correlation.csv（Pearson/Spearman/
  linregress，按 model×ptype×dist + ALL 分组）/ trial_embeddings_index.csv / run_status.csv；7 张图
  (matplotlib, 无 seaborn) + 诚实 15 段报告 + RUN_STATUS.md（失败/缺失明列，不伪装 complete）。
  embedding 存 npz，CSV 只放索引/hash/path。已接入 `code/run.py --summarize`。
- 新增 `scripts/slurm/{train_prototype_drift_gpu,summarize_prototype_drift_cpu}.sbatch` +
  `submit_prototype_drift_full.sh`（gpu2node + mi_torch_cu118，fail-fast if no CUDA，1 GPU/job，logs/slurm）。
- Smoke（subjects 1,2、eegnet、seed 0、3 epochs、CPU，隔离在 `outputs/experiments/prototype_drift_v1_smoke/`）：
  12 cells / 72 metric rows / 0 failed；used_target_labels_for_training 全 False、n_target_labels_used_for_training≡0、
  无 NaN/Inf、5 tables + 7 figures + report + RUN_STATUS 全生成 → 端到端通过。
- Full：50 eligible subjects（47×6 + 3×2 = 288 cells/(model,seed)）× 3 models × 5 seeds = 4320 cells，
  ~25920 metric rows。提交 15 GPU + 1 summarizer（afterany），job ids 21536-21551；eegnet 已 RUNNING，
  其余 PENDING(QOS CPU limit)，summarizer PENDING(Dependency)。
- 状态：**submitted/pending，未 complete**。完成判据：summarizer 跑完且 run_status.csv 全 ok。
- 检查命令：`squeue -u $USER`；`cat outputs/experiments/prototype_drift_v1/full_job_ids.txt`；
  完成后看 `4_experiments/prototype_drift/{report,tables,figures}/`。
- 边界守护：未触碰 /share/workspace2；未覆盖 Phase 0/1/2a/2b 结果；新结果只进 prototype_drift_v1 / 
  4_experiments/prototype_drift；不做 adaptation/online/memory/tool-routing/41-10/LOSO/fine-tune。
- Next：等 jobs 跑完 → 读 tables + REPORT 写 AI 分析（prototype drift 是否成立 / 分叉到 Step 4 or 其他机制）。

## 2026-06-11 — 汇总器迁入 code/ + canonical 9 段报告

- 新增 `code/summaries/{session,multisource,alignment}.py`：从 `backup/root_archive_2026-06-10/scripts/`
  迁入三个汇总器（无 src 依赖，仅修 PROJECT_ROOT=parents[2] 与 main(argv)）。
- 新增 `code/summaries/canonical.py`：按 AGENTS §8 的 9 段结构（Core conclusion / Goal / Method /
  Protocol / Results / Analysis / Relationship / Next step / File list），数据驱动地从汇总 CSV 生成
  `REPORT.md`，不捏造数字。
- 新增 `code/summaries/summarize.py` 调度器；`code/run.py` 增加 `--summarize`。
- 验证（真实数据）：用 backup 现成 30150 行 alignment run CSV 跑 Phase 2b 汇总 → 表/图/原始报告齐全，
  canonical 报告结论 BN-stats +0.0071、5 方法 Δacc 与历史一致。临时输出已清理。
- 用法：`python code/run.py --summarize --config code/configs/experiments/<phase>.yaml`。

## 2026-06-10 — 恢复新架构可训练能力（直连 code/，CPU smoke 通过）

- 新增 `code/runners.py`：把旧 `scripts/{run_session_drift,train_session_models,train_session_multisource,
  train_session_alignment}.py` 的逻辑改写为进程内 runner，全部从 `code.` 包导入，不再依赖归档脚本。
- 改写 `code/run.py`：去掉对归档脚本的 subprocess 路由与执行拦截，改为按 experiment 名直连
  `code.runners.PHASE_RUNNERS`；`--dry-run` 仍可用，新增 `--methods/--folds/--ckpt-dir/--tag-suffix` 等参数。
- CPU smoke 验证（真实数据）：
  - Phase 0 drift（sub-001/002，6 pairs）→ 生成 CSV/summary/报告/图，DONE。
  - Phase 1 within EEGNet（2 被试，2 折，1 epoch）→ 真训练，写出 runs CSV + meta + split，ALL DONE。
- 仍未迁移：summarize_* 汇总脚本仍在 `backup/root_archive_2026-06-10/scripts/`；训练能跑，结果聚合待迁移。
- 临时 smoke 输出已清理，未污染阶段目录已同步结果。

## 2026-06-10 — P10 风格重构收尾：结果同步 + 文档精简 + progress 恢复（无新实验）

- 把历史可读结果从 `backup/root_archive_2026-06-10/` 同步进新阶段目录：
  `1_session_drift/`（漂移报告/CSV/图）、`2_baseline/`（baseline + multi-source 报告/表/图）、
  `2_baseline/alignment_baseline/`（alignment 报告/表/图）。原始 per-run/checkpoint 仍在 backup。
- 恢复本文件 `progress.md`（来自 backup 的 PROGRESS.md），作为长期进度记忆。
- 精简 `0_docs`：合并为 `ARCHITECTURE.md`、`STATUS.md`、`FILE_CATALOG.md`、`operation_log.md`、`README.md`；
  删除 `STRUCTURE_AND_FILE_GUIDE.md`、`RUN_READINESS_AND_CLEANUP_GUIDE.md`、`CODE_ARCHITECTURE.md`、
  `PROJECT_STATUS_CURRENT.md`；操作日志移出 `01_Lab_Journal` 子目录到 `0_docs/operation_log.md`。
- 更新 `AGENTS.md` 权威文件表、目录路由、兼容说明、handoff 更新规则指向新文件名。
- 当前状态不变：Phase 0/1/2a/2b 已完成，Phase 2c Prototype Drift 为下一步、未运行。
- 运行就绪：`code/run.py --dry-run` 可用；完整训练需让 `code/run.py` 直连 `code/experiments`
  或临时从 backup 恢复 `scripts/src/configs`。

> **CURRENT STATUS (2026-06-08, Step 1 multi-source DONE):** The cross-session DG mainline
> now has: **A** session-drift diagnostic (144 pairs / 50 subjects) ✅; **B** static baseline
> EEGNet/DeepConvNet/FBCNet, 5 seeds, within-session 10-fold CV + single-source directed
> cross-session (26 520 trainings, 30/30 cells, no leakage/NaN) ✅; **Step 1 / C** the
> **multi-source `ses-01+ses-02 → ses-03`** static baseline ✅ (47 eligible subjects, 4 skipped,
> 705 rows, no NaN) — multi-source beats the best single source for all 3 models
> (EEGNet 0.7717 vs 0.7492, +0.0224; DeepConvNet 0.7564, +0.0353; FBCNet 0.6750, +0.0267;
> mean Δ +0.0281). Within Acc: EEGNet 0.807 / DeepConvNet 0.766 / FBCNet 0.720; single-source
> cross Acc 0.711 / 0.681 / 0.628 (drop 9–13%). We trail the paper by ~5–8pp in absolute within
> accuracy — a **training-recipe/data-budget** effect (within folds carve a 20% val slice →
> ~144 train trials/fold; dropout 0.25, no max-norm; shared un-tuned recipe), **not an
> architecture bug** (ranking + S1<S2<S3 trend match the paper).
> **Step 2 no-learning adaptation baseline = COMPLETE (2026-06-09).** 30,150 rows, 0 failed/0 NaN.
> Result is a **negative/diagnostic** one: no-learning alignment is INSUFFICIENT — none clears the
> +2pp line. none_reference acc 0.6818; best = bn_statistics_adaptation 0.6889 (Δ +0.0071); EA/RA
> slightly hurt; z-score/filter-bank ≈ neutral. High-drift subjects helped least → motivates (but
> does NOT run) learning-based Step-3 adaptation.
> **NEXT = Step 3 (learning-based) is FUTURE, NOT run.** CAP-EEGNet (v1/v2) + agent/toolkit/
> prototype/confidence/online/fine-tuning, LOSO, 41/10 remain **future work, not run**.
> GPU env `mi_torch_cu118` (torch 2.7.1+cu118).
> Sources of truth: `docs/PROJECT_STATUS_CURRENT.md` (status), `docs/MULTISOURCE_STEP1_REPORT.md`
> (Step 1), `docs/NEXT_EXPERIMENT_PLAN.md` + `docs/ADAPTATION_BASELINE_PLAN.md` (Step 2 plan),
> `docs/RESULTS_SUMMARY.md` (consolidated results), `docs/P10_INTEGRATION_SUMMARY.md` (P10 dir).


## 2026-06-10 — P10-style multi-dataset architecture reorganization (no experiments)

**Scope**: repository organization and documentation/rules only. No training, no preprocessing, no Slurm job, no external dataset write. `/share/home/yuan/SYX/P10_MI泛化研究` and `/share/home/yuan/SYX/CLAUDE.md` were used as read-only references.

**Changed**:
- Added P10-style root folders: `0_docs/`, `1_session_drift/`, `2_baseline/`, `3_online_adaptation/`, `4_experiments/`, `5_papers/`, `inbox/`, `01_Lab_Journal/`.
- Added modular `code/` framework with `run.py`, split configs (`datasets/models/experiments`), dataset adapters for WBCIC-SHU and SHU, channel mapping, and copied current model/method/experiment implementation for staged migration.
- Rewrote root memory files (`AGENTS.md`, `CLAUDE.md`, `README.md`, `proposal.md`, `results.md`, `experiment_log.md`) and rebuilt `.cursor/rules/` around the new architecture.
- Updated major Slurm entry scripts to call `code/run.py` with new phase configs. Existing outputs/checkpoints/logs and old `src/`/`scripts/` are retained for provenance/compatibility.

**Validation**: `python -B code/run.py --all --dry-run` passed; AST parsing passed for 44 files under `code/`; core imports passed inside `mi_torch_cu118` (`EEGNetClassifier`, `TrainSpec`, `ChannelZScore`).

<!-- AUTORUN_STATUS_BELOW: baseline_report.py inserts entries here -->

## 2026-06-09 — Anti-data-loss: backup/recovery scheme + GPU-cluster doc integration (no experiments)

**Why**: earlier loss happened because important code/docs lived only in the working tree, and
results (`outputs/`, `checkpoints/`, `logs/`) are gitignored so commits don't protect them. Built a
4-layer safety net + made the "soul files" + cluster knowledge durable. No experiment run.

- **Soul files committed/refreshed**: `AGENTS.md` + `.cursor/rules/{00-project-context,50-server-slurm,
  90-agent-behavior}.mdc` — generalized the senior-handoff-folder rule (no hard-coded P10 path; read &
  integrate when pointed at it), added explicit dataset addresses + project architecture/data-flow,
  tightened the filesystem-scope rule (write only inside the project root; `/share/workspace2` dataset
  is read-only; sibling handoff folders read-only), and corrected the GPU/Slurm guidance (env
  `mi_torch_cu118`, GPU jobs are normal now, fail-fast on no-CUDA).
- **GPU cluster doc integrated** (official `http://10.26.1.75:58080/`, "4090D 集群用户文档") into
  `docs/SERVER_RUNBOOK.md` + the 50-rule + AGENTS verified facts: 1 login01 + 5 GPU nodes
  (gpu01-05, 8× 4090D, no SSH, Slurm-only) + storge file node; partitions gpu2node(gpu01,02)/
  gpu3node(gpu03,04,05); monitoring `slmwatch`/`gpuwatch`/`user-tools`; modules; home quota 512 GiB;
  IO-heavy work on storge.
- **New doc `docs/BACKUP_AND_RECOVERY.md`**: what git manages vs. gitignored, the 4 layers
  (commit → tag → `git bundle` → results tar), where backups live (authorized
  `/share/home/yuan/SYX/backups/`; large → `/share/workspace2`), restore steps, and a
  before/during/after routine checklist.
- **Milestone tagged** `milestone-step2-alignment-complete` and **backups created** under
  `/share/home/yuan/SYX/backups/`: `eeg-mi-online_git_2026-06-09.bundle` (full git history+tags) +
  `eeg-mi-online_results_2026-06-09.tar.gz` (session_drift_v1 + baseline_v1 + alignment_baseline_v1
  CSVs/reports/figures/splits/configs/job-ids; the 11 GB alignment checkpoints intentionally NOT
  archived — re-creatable) + `BACKUP_MANIFEST.md` (sizes + sha256). Working tree clean.

## 2026-06-09 — Step 2 no-learning alignment baseline COMPLETE (negative/diagnostic result)

**Status: COMPLETE.** All 75 GPU training jobs (`21261-21335`) + the summarizer (`21336`) finished
COMPLETED; `results_alignment_all.csv` exists with **30,150 rows** (25,125 alignment + 5,025
none_reference), **0 failed**, **0 NaN/Inf** in metrics (the only all-empty column is the text
`error_message`). `used_target_y_for_training == False` for all rows; `used_target_x_for_stats ==
True` for all 25,125 trained-method rows. 5 trained methods + none_reference all present; models
eegnet/deepconvnet/fbcnet; seeds 0-4; all 6 single-source directions + multi-source ses-01+02→ses-03.

**Result (honest, mean cross-acc over directions/subjects/seeds, vs baseline_v1 `none_reference`):**

| method | Δacc vs none | note |
|---|---:|---|
| `bn_statistics_adaptation` | **+0.0071** | only net-positive method; well below +0.02 |
| `filterbank_reweighting` | −0.0030 | ≈ neutral |
| `session_zscore` | −0.0038 | ≈ neutral |
| `riemannian_alignment` | −0.0101 | slightly hurts |
| `euclidean_alignment` | −0.0124 | hurts most |

- Absolute acc (all scope): none_reference **0.6818**; bn_statistics **0.6889** (best); all others < none.
- By drift level: BN-stats small-positive at every level (stable +0.009 / moderate +0.008 / high
  +0.005); filter-bank positive on stable/moderate (+0.011/+0.005) but **−0.024 on high drift**;
  EA/RA negative everywhere. → the subjects that drift most are helped least.
- **Conclusion: no-learning / unsupervised statistic-only alignment is INSUFFICIENT** (no method
  reaches the pre-registered +2pp success line). BN-stats gives only a small positive gain. This is a
  **valuable negative/diagnostic result** that objectively justifies learning-based Step-3 adaptation
  (online / adapter / prototype / memory) — **which is NOT run here**.
- Verified the `alignment_vs_baseline.csv` join is correct: none_reference is unique per
  (model, seed, subject, train_sessions, test_session, training_scope); all 25,125 alignment rows
  matched; an independent re-merge reproduced the summarizer's Δacc exactly.
- Outputs: `outputs/experiments/alignment_baseline_v1/` — `ALIGNMENT_BASELINE_REPORT.md` (13 sections
  + headline), `RUN_STATUS.md`, `manifest_sources.json`, `cross_session/tables/` (9 CSVs incl.
  results_alignment_all.csv, alignment_vs_baseline.csv, alignment_by_method/model/direction/protocol/
  subject.csv, alignment_gain_by_drift_level.csv, run_status.csv), `cross_session/figures/` (6 PNGs).
  Checkpoints `checkpoints/alignment_baseline_v1/`.
- Closeout only added a richer, honest interpretation to the report (recomputed from the same CSVs);
  no new experiment, no new deps, baseline_v1 untouched.

## 2026-06-09 — Step 2 no-learning alignment baseline: implemented + smoke + FULL RUN SUBMITTED (results PENDING)

**Status (do NOT read as complete):** Step 2 **code implemented**, **smoke passed**, **full run
submitted**, **summarizer dependency submitted**, **results PENDING until the 75 GPU jobs finish.**
Will be marked complete only after the summarizer writes `results_alignment_all.csv`.

**Scope (per user): ONLY Step 2 alignment baseline.** No online, no 41/10, no LOSO, no fine-tuning,
no CAP-EEGNet full, no multi-agent/prototype/memory. No new deps, no shared-env changes, no
raw/workspace2 writes, baseline_v1 not overwritten.

**A — pre-checks (all pass):** baseline_v1 exists with within+cross standard tables; processed
manifest + `eog_ecg_clean` present; env `mi_torch_cu118` = torch 2.7.1+cu118, CUDA available on the
GPU node (RTX 4090 D); git working tree clean (HEAD `a239b43`).

**B — new code (nothing existing overwritten; reuses trainer / registry / session_splits / metrics
/ baseline split logic):**
- `src/adaptation/{__init__,session_alignment,bn_adaptation}.py` — fit/transform alignment
  transforms (ChannelZScore, EuclideanAlignment, RiemannianAlignment = log-Euclidean SPD mean via
  eigh, FilterBankReweight) + BN running-stat adaptation (no grad/backward/optimizer). EA/RA use
  eps ridge + diagonal shrinkage; inverse-sqrt/logm via symmetric eigh with eigenvalue clipping.
- `src/evaluation/session_alignment_protocols.py` — single-source + multi-source tasks; source
  transform fit on SOURCE TRAIN only (applied to source train+val); target aligned from target X
  (filterbank reweights target→source profile); BN method trains plain then refreshes BN from
  target X; per-run leakage asserts; 26-column result rows.
- `scripts/train_session_alignment.py`, `scripts/summarize_alignment_results.py` (pulls
  `none_reference` from baseline_v1 + drift levels), `scripts/build_alignment_baseline_outputs.py`
  (scaffold). `configs/session_alignment_compare.yaml`. Slurm:
  `scripts/slurm/{train_session_alignment_gpu,summarize_alignment_results_cpu}.sbatch`.

**C — unit checks (CPU, synthetic):** zscore finite + ~N(0,1); EA matrix 58×58; RA matrix 58×58 +
spd_mean=log_euclidean; aligned X shape unchanged; EA finite on near-singular cov; BN adapt leaves
all trainable weights UNCHANGED (no optimizer.step) but changes running_mean; eval mode after; all 3
models forward aligned X → logits [B,2]. ALL PASS.

**D — smoke (GPU srun, mi_torch_cu118):** subjects 1,2, eegnet, seed 0, 3 epochs, all 5 trained
methods, both protocol groups → 70/70 rows ok, 0 NaN/Inf, n_train/val/test = 160/40/200 (single) &
320/80/200 (multi), 70 checkpoints, split JSON sessions correct, `used_target_y_for_training`=False
& `used_target_x_for_stats`=True for all rows. Summarizer ran cleanly on the smoke rows. Smoke
artifacts removed afterward.

**E — full run (GPU Slurm):** **75 jobs `21261–21335`** = method × model × seed (each job runs both
single-source + multi-source). Per job = 335 trainings; total ≈ **25,125 trainings**. Partition
gpu2node, gpu:1, env mi_torch_cu118, fail-fast if no CUDA, logs → `logs/slurm/`. Job ids →
`outputs/experiments/alignment_baseline_v1/full_job_ids.txt`. (Some pending under
QOSMaxCpuPerUserLimit; drain as capacity frees.) Estimates + layout in
`outputs/experiments/alignment_baseline_v1/RUN_PLAN.md`.

**F — dependent summarizer:** CPU job **`21336`** with `--dependency=afterany:<all 75>` →
`scripts/summarize_alignment_results.py` → `cross_session/tables/*` (results_alignment_all.csv,
alignment_by_method/model/direction/protocol/subject.csv, alignment_vs_baseline.csv,
alignment_gain_by_drift_level.csv, run_status.csv) + `cross_session/figures/*` (6 figs) +
`ALIGNMENT_BASELINE_REPORT.md` + `RUN_STATUS.md` + `manifest_sources.json`.

**Next (after jobs finish):** check `sacct -j 21261-21336`, read `RUN_STATUS.md`, confirm
`results_alignment_all.csv` exists, then read `ALIGNMENT_BASELINE_REPORT.md`. Only then mark Step 2
complete. Step 3 (online / adapter / prototype) stays gated on these results.

## 2026-06-09 — Systematic documentation repair (no experiments)

**Why**: git HEAD is the 2026-06-04 scaffold commit; all work since (drift, baseline, P10
integration, Step 1) was never committed, so the working tree's docs/code lived only on disk.
During the Step-1 run a tooling/backend hiccup left the working tree missing several docs and
the new Step-1 code files. **Experiment results were never lost** (the 705-row Step-1 run +
the baseline summaries were intact on disk).

**Repaired (no results touched, no experiments run):**
- Re-materialized the Step-1 code that the run had used: `src/evaluation/session_multisource_protocols.py`,
  `scripts/train_session_multisource.py`, `scripts/summarize_multisource_results.py`,
  `configs/session_multisource_compare.yaml`, the two `scripts/slurm/*multisource*` jobs.
  Re-ran the summarizer on the existing 705 rows — reproduced identical numbers.
- Restored the full rich docs from user-provided backups (`/share/home/yuan/SYX/PROGRESS.md`,
  `/share/home/yuan/SYX/PROJECT_OVERVIEW.md`, "saved before the remaining Step 1") + this
  conversation's context, then brought them current (P10 integration + Step 1 done).
- Rewrote the lost docs: `BASELINE_PROTOCOL.md`, `P10_INTEGRATION_SUMMARY.md`,
  `ADAPTATION_BASELINE_PLAN.md`, `CODE_INTEGRATION_NOTES.md`,
  `references/P10_MI_generalization_README.md`; recovered+updated `AGENTS.md`,
  `references/README.md` from git HEAD.
- Wrote a deep Step-1 analysis: `docs/MULTISOURCE_STEP1_REPORT.md` (per-subject gains,
  drift-level breakdown, upper-bound gap recovery) + `multisource_by_subject.csv`.

**Recommendation**: commit the working tree soon so this can't recur (currently everything since
2026-06-04 is uncommitted). Not done automatically — awaiting user go-ahead on git.

## 2026-06-08 — Step 1 static baseline COMPLETE: multi-source ses-01+02 → ses-03

**Scope (per user): only the remaining Step-1 static-baseline item.** No Step-2 adaptation,
no online, no 41/10, no LOSO, no fine-tuning, no CAP-EEGNet full / multi-agent / prototype /
memory. No raw/workspace2 writes. No existing baseline results overwritten.

**Step A — checked for existing results**: none found under `outputs/experiments/` for
`multisource` / `0102_to_03`. Confirmed in code that only **single-source directed pairs**
existed (`session_splits.make_cross_session_pairs`, `session_protocols.run_cross_session`).
So the multi-source direction was genuinely missing → implemented it.

**Step B — new code (separate from existing baselines, nothing overwritten):**
- `src/evaluation/session_multisource_protocols.py` — eligibility (all of ses-01/02/03 ok),
  combine ses-01+ses-02 trials, carve val ONLY from the combined train, test = ses-03;
  per-run leakage assertions; `n_train/n_val/n_test` + checkpoint + status recorded.
- `scripts/train_session_multisource.py` — entry (reuses TrainSpec / trainer / registry /
  `load_ok_sessions`); writes per-seed CSV + meta JSON (used/skipped subjects).
- `scripts/summarize_multisource_results.py` — aggregates, builds the report + figure, and
  pulls the single-source ses-01→03 / ses-02→03 rows from the baseline summaries for comparison.
- `configs/session_multisource_compare.yaml`, `scripts/slurm/train_session_multisource_gpu.sbatch`,
  `scripts/slurm/summarize_multisource_results_cpu.sbatch`.

**Step C — smoke test (GPU node via srun, `mi_torch_cu118`):** subjects 1,2, EEGNet, seed 0,
3 epochs. Passed: train_sessions=ses-01+ses-02, test=ses-03, `n_train=320 / n_val=80 / n_test=200`,
train/val disjoint, val carved only from the combined train, no NaN, CSV + checkpoint + split JSON
written. Smoke output dir removed afterward.

**Step D — full run (GPU Slurm):** jobs `21240,21241,21242,21243,21244` (one per seed,
3 models each) + summarize `21245` (afterany). All **COMPLETED**, exit `0:0`. 705 rows
(3 models × 5 seeds × 47 used subjects), 0 failed, 0 NaN.

**Step E — results (mean±std over 5 seeds, test = ses-03):**
- EEGNet **0.7717±0.003** > DeepConvNet 0.7564±0.007 > FBCNet 0.6750±0.002.
- vs best single direction (`ses-02→03`): +0.0224 / +0.0353 / +0.0267 (mean **+0.0281**).
- vs per-subject best single (oracle source pick): +0.0139 / +0.0209 / +0.0097 (still positive).
- recovers ~30–53% of the single-source-cross → within-ses-03 upper-bound gap.
- per-subject: EEGNet improved 33/47; failures concentrate where the two source sessions differ
  a lot (e.g. sub-029 ses-01 0.441 vs ses-02 0.935) → motivates Step-2 alignment/weighting.
- Full analysis: `docs/MULTISOURCE_STEP1_REPORT.md`; machine report + CSVs under
  `outputs/experiments/baseline_v1/provenance/session_multisource_v1/summaries/`; checkpoints
  `checkpoints/session_multisource_v1/{eegnet,deepconvnet,fbcnet}/`.

**Note (doc loss + recovery):** during this work the (uncommitted) working tree lost several
docs and the new Step-1 code files (git HEAD is the 2026-06-04 scaffold; later work was never
committed). The experiment **results were intact**; the Step-1 code was re-materialized from the
run, the summarizer re-ran cleanly on the existing 705 rows, and the docs were restored from
backups + this conversation's context. See the 2026-06-09 doc-repair entry.

**Next**: Step 2 no-learning adaptation baseline (see `docs/ADAPTATION_BASELINE_PLAN.md`). Do not
run online / 41-10 / fine-tuning / CAP-EEGNet full until Step 2 is done.

## 2026-06-08 — P10 「MI 泛化研究」 integration + direction cleanup (NO experiments run)

Integrated the senior's updated P10 package (`/share/home/yuan/SYX/P10_MI泛化研究/`) into the
project's understanding/docs. P10 Phase-0 (drift) + Phase-1 (baseline) numbers **match this
project exactly** (same 148 ok sessions, same EEGNet 0.807/0.711) — same study, packaged for
handoff. P10 adds richer per-subject drift profiling (drift_level high/moderate/stable; sub-020
most drifted, sub-005 most stable), the cross-session **directional asymmetry** (ses-02→03 0.749
best vs ses-03→01 0.681 worst, 6.8pp), MMD↔accuracy Spearman ρ=−1.0 (n=3), and a **Phase-2 online
continual-learning framework that is design + draft code only, NOT validated** → future work.

New docs: `docs/PROJECT_STATUS_CURRENT.md`, `docs/P10_INTEGRATION_SUMMARY.md`,
`docs/NEXT_EXPERIMENT_PLAN.md` (Step 0/1/2/3), `docs/ADAPTATION_BASELINE_PLAN.md`,
`docs/CODE_INTEGRATION_NOTES.md`, `docs/references/P10_MI_generalization_README.md`. Updated
wording (done=done, unrun=future) across the core docs. **No experiment run; no raw/workspace2 writes.**

<!-- AUTORUN_STATUS_BELOW: baseline_report.py inserts entries here -->

## 2026-06-07 — Baseline 5-seed run COMPLETE + repo cleanup

**All 10 training jobs + the report job COMPLETED** (21161–21171, exit 0). Final summarizer
re-run after cleanup: **26 520 rows (within 22 200 + cross 4320), 30/30 cells,
`incomplete=False`, no NaN, no leakage**. (The earlier auto `RUN_STATUS.md`/log entry said
"INCOMPLETE" only because the report job counted *itself* (21171) as RUNNING — a false flag;
the summarizer logged `incomplete=False`.)

**Headline results** (mean±std over 5 seeds):
- within Acc: EEGNet **0.807±0.002** > DeepConvNet 0.766±0.002 > FBCNet 0.720±0.003.
- cross Acc: EEGNet **0.711±0.008** > DeepConvNet 0.681±0.002 > FBCNet 0.628±0.003.
- cross-session drop: 11.9% / 11.1% / 12.8%.
- vs paper within (85.32 / 84.47 / 78.40): we are −4.65 / −7.84 / −6.37 pp.

**Why below paper (see `docs/RESULTS_SUMMARY.md` §4)**: systematic training-recipe/data-budget
difference, NOT an architecture bug — chiefly the within-fold 20% val carve (≈144 vs 180 train
trials), early stopping on a tiny (~36) val set, dropout 0.25 + no max-norm + wd 0, and a shared
un-tuned recipe (FBCNet also simplified). Gap is consistent across all 3 models and the ranking
+ S1<S2<S3 trend match the paper; the within-vs-cross conclusion is fair and unaffected.

**New doc**: `docs/RESULTS_SUMMARY.md` (drift + baseline + paper-gap analysis + next steps).

**Repo cleanup (gitignored artifacts only; no source/results lost):** removed the archived
smoke output dir, legacy `outputs/processed_paper_style/sub-001`, `outputs/raw_check`,
`outputs/sanity_check`, and the stale smoke-only `checkpoints/.../cap_eegnet/` (18 files,
sub-001/002 seed0). Standardized run CSV naming: legacy `within__{model}.csv` →
`within__{model}__seed0.csv` (now all 30 files are uniform `{protocol}__{model}__seed{N}.csv`).
The 3 baseline checkpoint dirs (eegnet/deepconvnet/fbcnet, 2180 each, ~2.7G total) are kept as
legitimate results.


## 2026-06-07 — Overnight 5-seed baseline GRID submitted (EEGNet/DeepConvNet/FBCNet)

**Goal**: full 5-seed baseline over the 148 `status=ok` sessions, both protocols, seeds
0–4, three baseline models. CAP-EEGNet and all complex modules remain FUTURE (not run);
no LOSO, no 41/10.

**Overwrite fix**: result CSVs are now written per seed —
`runs/{protocol}__{model}__seed{seed}.csv` (was `{protocol}__{model}.csv`, which would have
let seeds overwrite each other). Job 21161 (already running, old code) keeps the legacy
`within__{model}.csv` (=seed 0); the summarizer globs all CSVs and reads the `seed` column,
so there is no collision. Checkpoints were already seed-namespaced.

**Jobs (1 GPU each, env `mi_torch_cu118`, fail-fast if no CUDA):**
- `21161` within seed 0 (already running) — `mi_sess_compare`
- `21162` cross s0, `21163` within s1, `21164` cross s1, `21165` within s2, `21166` cross s2,
  `21167` within s3, `21168` cross s3, `21169` within s4, `21170` cross s4
  (job names `mi_base_{protocol}_s{seed}`, logs `logs/slurm/mi_base_{protocol}_s{seed}-<jobid>.out/.err`)
- job-id list: `outputs/experiments/baseline_v1/provenance/session_model_compare_v1/overnight_job_ids.txt`

**Expected trainings** (1 model × 1 protocol × 1 seed): within = 148 sessions × 10 folds =
1480; cross = 288 directed pairs. Per seed (3 models): within 4440, cross 864. 5-seed total:
within 22 200 + cross 4320 = **26 520 trainings**.

**Dependent report job** (`mi_base_report`, CPU, `--dependency=afterany:` all 10 IDs): runs
`scripts/baseline_report.py` → sacct check + `scripts/summarize_session_results.py` →
`outputs/experiments/baseline_v1/provenance/session_model_compare_v1/summaries/` (results_within/cross_session.csv,
within_by_seed/cross_by_seed/within_session_wise/cross_by_direction.csv,
summary_by_model_protocol.csv, model_ranking.md, SESSION_MODEL_COMPARE_REPORT.md, RUN_STATUS.md,
3 figures) and appends auto status entries here + to `docs/EXPERIMENT_LOG.md`. Marks the
report INCOMPLETE if any training job is not COMPLETED or any output is missing.

Outputs: `outputs/experiments/baseline_v1/provenance/session_model_compare_v1/`; checkpoints
`checkpoints/session_model_compare_v1/<model>/{within_<subj>_<ses>_seed<k>.pt,
cross_<subj>_<tr>-to-<te>_seed<k>.pt}`; slurm logs `logs/slurm/`.

## 2026-06-07 — Mainline converged to the three baseline architectures (EEGNet/DeepConvNet/FBCNet)

**Decision (per user)**: narrow the active mainline to **the three baseline models** — run
EEGNet + DeepConvNet + FBCNet within-session 10-fold CV then cross-session on the 148
`status=ok` sessions; start by checking EEGNet against the WBCIC-SHU paper baseline.
**CAP-EEGNet (v1/v2) and ALL complex modules (agent/toolkit/neural-subagents, prototype,
multi-source confidence, online, fine-tuning) are kept in code but marked OPTIONAL /
FUTURE — not run.** Also not run: LOSO, 41/10.

**No code deleted.** Changes are config + docs only (the trainer/protocols stay
model-agnostic, so CAP-EEGNet still works when re-enabled):
- `configs/session_model_compare.yaml`: `models: ["eegnet","deepconvnet","fbcnet"]`
  (cap_eegnet commented as optional/future).
- `scripts/slurm/train_session_models_gpu.sbatch`: default `--models eegnet,deepconvnet,fbcnet`.
- Docs updated: `PROGRESS.md` (this entry + top status), `EXPERIMENT_PROTOCOL.md`
  (mainline = 3 baselines; CAP-EEGNet + experiments = future), `BASELINE_PROTOCOL.md`
  (top banner + run commands narrowed to the 3 baselines).

**Run plan (seed=0)**: (1) full within-session CV for the 3 baselines → (2) summarizer →
check EEGNet vs paper → (3) full cross-session for the 3 baselines → summarizer.
Submitting the full within-session CV now (see next entry / experiment log for results).

## 2026-06-06 — Mainline pivot: cross-session DG (drift diagnostics + within/cross model comparison)

**Why**: per the senior's P10 "MI 泛化研究" package (now in `docs/references/P10_MI_generalization/`),
the project pivots from 41/10 cross-subject pretraining to a **cross-session domain
generalization** study: (A) diagnose what drifts across sessions, (B) compare baselines +
our model under one fair protocol, (C) at Within-session CV and Cross-session levels, (D)
unified report. **No 41/10, LOSO, fine-tuning, or online now** — all kept as future work.

**Task 0 — references placed** (renamed, reference-only, not run from there):
`docs/references/P10_MI_generalization/` (HANDOFF/proposal/experiment_log/QC_SUMMARY_CN/
ChatGPT task md/paper PDF/slides) and `docs/references/senior_scripts/{data_validation,
model_training}/` (the originals). `docs/references/README.md` indexes them + maps each to
its in-tree implementation.

**New configs**: `configs/session_drift.yaml`, `configs/session_model_compare.yaml` (paths
from `configs/paths.yaml`; status=ok filter; within folds=10; protocols within+cross;
models eegnet/deepconvnet/fbcnet/cap_eegnet; seeds=[0]; bs16; lr1e-3; max_epochs100;
early-stopping).

**New code**:
- `src/data/session_splits.py` — `load_ok_sessions` (manifest, status=ok), label
  normalization to {0,1} (accepts {1,2}), `make_within_session_folds` (StratifiedKFold,
  label-balanced), `make_cross_session_pairs` (directed, both-ok), JSON persistence.
- `src/analysis/session_drift.py` + `scripts/analysis/run_session_drift.py` +
  `scripts/slurm/session_drift_cpu.sbatch` — MMD/CORAL/μ-β power shift/KS/ERD-ERS spatial
  corr/CSP similarity/RMS ratio/Fisher shift; vectorized FFT; **matplotlib-only** (no
  seaborn); status=ok via manifest; `--subjects`/`--max-subjects`. Outputs CSV + summary.json
  + SESSION_DRIFT_REPORT.md + 8 figures.
- `src/models/`: added `EEGNetClassifier` (eegnet.py), `deepconvnet.py` (Schirrmeister
  2017), `fbcnet.py` (Mane 2021, fixed FIR filter bank), `registry.py` (`build_model`).
  **CAP-EEGNet upgraded to v1**: encoder + classifier + lightweight **learned** confidence
  head (single-source scalar, calibration BCE); prototype/subagents/adapter/domain/online
  still raise `NotImplementedError`. All 4 share `forward -> {logits, features, confidence}`.
- `src/training/trainer.py` (CE + optional confidence-BCE, early stopping, predict),
  `src/evaluation/session_protocols.py` (`run_within_session` StratifiedKFold; 
  `run_cross_session` directed pairs; metrics acc/bacc/f1/auc/nll/brier/ece; val carved
  from train only; bounded checkpointing), `scripts/train_session_models.py` +
  `scripts/slurm/train_session_models_gpu.sbatch` (env `mi_torch_cu118`, cuda fail-fast).
- `scripts/summarize_session_results.py` + `..._cpu.sbatch` — merges `runs/*.csv` →
  results_within/cross CSVs, summary_by_model_protocol.csv, 3 figures, model_ranking.md,
  SESSION_MODEL_COMPARE_REPORT.md, cross-session drop + relative drop.

**Docs**: new `SESSION_DRIFT_ANALYSIS.md`, `BASELINE_PROTOCOL.md`; updated
`PROJECT_OVERVIEW.md` (top status), `EXPERIMENT_PROTOCOL.md` (mainline = within/cross;
41/10/LOSO/finetune/online = FUTURE), `.cursor/rules/30-model-experiments.mdc` (shared
protocol/filter/metrics + no-leakage), `AGENTS.md` (constraint 9).

**SMOKE TESTS — ALL PASS (compute nodes via srun, never login node):**
- GPU env check: `mi_torch_cu118` = torch **2.7.1+cu118**, `cuda.is_available()=True`,
  RTX 4090 D. (The earlier `mi_torch` CPU-only-torch blocker is resolved by this env.)
- Drift (subjects 1,2): 6 within-subject pairs in 12.6 s; CSV + summary.json + report +
  8 figures written to `outputs/analysis/session_drift_v1/`. MMD≈0.175, ERD/ERS corr≈0.44–0.49,
  RMS ratio≈0.90 — sane.
- Within (subjects 1,2; folds=2; epochs=3; all 4 models): 48 rows, 117.8 s, all CSVs +
  per-session split JSONs written.
- Cross (subjects 1,2; epochs=3; all 4 models): 48 rows (2 subj × 6 directed pairs × 4
  models), 37.7 s; n_train=160/n_val=40 (carved from train) / n_test=200 (full test session)
  → no leakage verified.
- Summarize: merged 96 rows → all 8 summary artifacts. (Numbers are smoke-noise at 3
  epochs/2 folds; pipeline correctness is the point.)

**Full-run estimates (1 seed; pending user go-ahead — NOT yet submitted):**
- Drift full ≈ **144** within-subject pairs (47 subjects×3 + 3 from partial-ok subjects;
  sub-024 has 1 ok session → 0 pairs), ~10–15 min CPU.
- Within full = 148 ok sessions × 10 folds × 4 models = **5920** trainings; ~**4–8 h wall**
  if the 4 models run as parallel 1-GPU jobs (deepconvnet is the long pole), less with
  early stopping.
- Cross full = **288** directed pairs × 4 models = **1152** trainings; ~**1–2 h wall**
  parallel. Seeds multiply linearly.
- Outputs: `outputs/analysis/session_drift_v1/`, `outputs/experiments/baseline_v1/provenance/session_model_compare_v1/`
  (`runs/`, `splits/`, `summaries/`); checkpoints `checkpoints/session_model_compare_v1/`;
  slurm logs `logs/slurm/`.

**Deliberately NOT done**: full-scale GPU submission (awaiting user OK per task 8.6); 41/10,
LOSO, fine-tuning, online; full multi-subagent CAP-EEGNet (v1 only). No raw/workspace2 writes.

## 2026-06-05 — Roadmap alignment (project direction = full multi-subagent CAP-EEGNet)

**Why**: ensure the project targets the chat record's FINAL design, not a plain EEGNet
classifier. No training, no GPU install, no sbatch — docs/rules/light-code alignment only.

**Final goal restated everywhere**: **Confidence-aware Online Adaptive Multi-Subagent
Pretraining Framework for Cross-subject MI EEG Decoding**. The minimal EEGNet+classifier is
explicitly a **Stage 0 pipeline-validation baseline**, NOT the paper method.

**Staged route fixed (docs/ROADMAP.md)**: Stage 0 infra/sanity (✅) → Stage 1 cross-subject
pretrain + zero-shot (5 seeds, one model each, mean±std) → Stage 2 confidence+prototype+
adapter(+domain, +neural subagents) → Stage 3 Session-1 fine-tune → Stage 4 online
test-then-update → Stage 5 ablation/interpretability.

**New docs**: `docs/ROADMAP.md` (final goal, full module table, minimal-vs-full, Stage 0–5,
do-not list) and `docs/ALIGNMENT_CHECKLIST.md` (chat-record→status mapping).

**Docs updated**: `PROJECT_OVERVIEW.md` (top status, §0 final-goal, completion table now
shows splits ✅ + minimal CAP-EEGNet ✅ + full CAP-EEGNet 🚧, next-steps), `MODEL_PLAN.md`
(minimal-vs-full, deep neural subagents not handcrafted, full component list, Stage-ordered
build order), `EXPERIMENT_PROTOCOL.md` (5-seed mean±std, Exp1–4 input/output/forbidden).

**Rules updated**: `00-project-context` (final goal + refreshed current-stage block),
`30-model-experiments` (minimal≠final; repeated multi-seed split mandatory; no target
leakage; stage note), `40-online-learning` (test-then-update; default-forbid full-backbone
online update; confidence-gate/prototype/adapter are core).

**Light code alignment** (`src/models/cap_eegnet.py`): kept the working minimal model;
added full-vision config flags (`use_subagents/use_dataset_router/use_adapter/use_prototype/
use_confidence/use_domain_align/use_online_update`), fail-fast `NotImplementedError`
("Reserved for full CAP-EEGNet … NOT implemented in the minimal sanity model") when any is
enabled, `predict_confidence()`/`online_update()` reserved methods, and documented stub
classes (`NeuralSubagentEncoder/DatasetAwareRouter/Adapter/PrototypeMemory/ConfidenceHead/
DomainAlignmentHead/OnlineUpdateModule`). Verified: minimal forward still returns
`{logits[B,2], features, proto_dist=None, confidence=None}`; enabling a full flag raises the
clear error. Also aligned `configs/finetune.yaml` + `configs/online_adaptation.yaml`
`variant→eog_ecg_clean` + `statuses:[ok]` for consistency with `train_cross_subject.yaml`.

**Unchanged on purpose**: preprocessing, splits, dataset, sanity outputs (no refactor). No
formal training, no GPU env changes, no sbatch, derivatives `.mat` never a training entry.

## 2026-06-05 — Pre-training prep: 41/10 splits + minimal CAP-EEGNet + sanity train

**Scope (per user): training-prep only, NOT formal training.** All heavy/GPU steps ran on
a compute node via `srun` (node gpu02), never the login node.

**1. `src/data/splits.py` (implemented).** Reads `processed_manifest.csv`, uses only
`status==ok` sessions. `SubjectSessionIndex` summarizes subject→session→status. Policy:
target subjects must have **all 3 sessions ok** (needed for Session-1 fine-tune +
Session-2/3 online); source = the other 41 subjects (may contain subjects with failed
sessions, but training uses ok sessions only). Subject-wise only — never session/trial.
Failed sessions → `excluded_sessions`. `make_subject_wise_split(seed=...)` uses a local
`random.Random(seed)` over the **47 fully-ok** subjects to pick 10 targets; the 4 partial
subjects (sub-023/024/026/032, owning the 5 failed sessions) are always forced into source.

**2. Splits generated** via `scripts/make_splits.py` for seeds **2026–2030** →
`splits/cap_eegnet_4110_seed<k>.json`. Each: source=41 (118 ok sessions, 23 600 trials,
11 800/11 800), target=10 (30 ok sessions, 6 000 trials, 3 000/3 000), excluded=5
(`sub-023/ses-01, sub-024/ses-02, sub-024/ses-03, sub-026/ses-01, sub-032/ses-02`). Each
JSON records source_subjects, target_subjects, excluded_sessions(+detail), source/target ok
sessions, `target_finetune_sessions=[1]`, `target_online_sessions=[2,3]`. Target sets per
seed (e.g. 2026: 007/008/015/021/030/037/040/043/046/049).

**3. Split smoke test** `tests/test_splits.py` (manifest-only, login-safe): asserts
source=41/target=10, no source∩target overlap, target subjects have ses-01/02/03 all ok,
all used sessions are ok, excluded are truly non-ok. **All 5 splits PASS**; prints per-split
trial/session counts + balanced 0/1 label dist.

**4. Minimal CAP-EEGNet** — `src/models/eegnet.py` now has a real `EEGNetEncoder`/`EEGNet`
(Lawhern 2018, F1=8/D=2/F2=16/K=64; depthwise spatial over 58 ch, separable temporal;
feature_dim inferred = 496). `src/models/cap_eegnet.py` `CAPEEGNet` (nn.Module) = encoder +
linear cls head; `forward(x)` accepts `[B,58,1000]` or `[B,1,58,1000]`, returns
`{logits[B,2], features[B,496], proto_dist=None, confidence=None}`. Adapter / Prototype /
Confidence kept as **TODO stubs** (use_*=False; enabling raises NotImplementedError). ~3 026
params. (`padding='same'` even-kernel UserWarning is benign.)

**5. SHUTrialDataset smoke test** `tests/test_dataset_smoke.py` (compute node): builds
source/target datasets from `processed_manifest.csv` + split JSON; asserts entry is `.npz`
only (no `derivatives/.mat`), item x=`[58,1000]` float32, DataLoader batch x=`[B,58,1000]`,
y dtype `torch.long` (int64), labels ⊆ {0,1}; feeds a batch through CAP-EEGNet → logits
`[B,2]` (internal 4D `[B,1,58,1000]`). **PASS** for both roles.

**6. CUDA check (srun, --gres=gpu:1 on gpu02)**: `nvidia-smi` shows an RTX 4090 D (24 GB,
driver CUDA 12.8), but **torch 2.6.0 is CPU-only**: `torch.version.cuda=None`,
`torch.cuda.is_available()=False`, `device_count=0`. **Fix plan (do NOT mutate shared
`mi_torch`)**: create a dedicated CUDA env, e.g. inside a GPU `srun`:
`conda create -n mi_torch_cu118 --clone mi_torch` then
`pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio`
(cu121/cu124 also compatible with the 12.8 driver), and point GPU jobs at it. Awaiting user
go-ahead before installing.

**7. Sanity training** `scripts/sanity_train.py` (+ `scripts/slurm/sanity_train_gpu.sbatch`):
3 source subjects (sub-001/002/003), 9 npz, 1 800 trials, 3 epochs, batch 128, Adam lr 1e-3.
Ran on CPU (no CUDA). **loss 0.7002 → 0.6474 → 0.6014 (decreased=True)**, acc 0.541 → 0.630 →
0.682, no shape errors, ~38 s. Output: `outputs/sanity_check/sanity_check_metrics.json`.
Device auto-selects cuda if available else cpu, so it will use GPU once cu-torch is installed.

**Deliberately NOT done**: formal cross-subject training loop, fine-tuning, online loops,
prototype/confidence/adapter heads, GPU torch install. Awaiting user go-ahead.

## 2026-06-05 — Full preprocessing QC + quality comparison vs official derivatives

**What ran**: full `eog_ecg_clean` preprocessing was already complete (153 sessions, written
to the external `processed/eog_ecg_clean/` tree). This step did **QC + a quality comparison
against the paper `derivatives/2C dataset_processeddata` .mat** (NO training, NO 41/10, NO
CAP-EEGNet). New code:
- `src/evaluation/data_quality.py` — reusable, numpy/scipy-only metrics: amplitude, Welch
  PSD + bandpowers (δ/θ/μ/β/low-γ/50 Hz), per-channel RMS/std, paired similarity
  (trial-wise + per-channel Pearson, MAE/RMSE/relRMSE), MI class separability (Cohen's d /
  Fisher on log-bandpower at C3/C4), official `.mat` locator (handles the one non-`eeg/`
  layout) + loader (data[58,1000,200]→[200,58,1000], labels 1/2→0/1), and
  `best_session_assignment()` (per-subject session matching).
- `src/visualization/quality_plots.py` (NEW package) — 13 QC/compare figures (Agg backend).
- `scripts/compare_processed_quality.py` — orchestrator: manifest QC (19 checks) →
  per-subject session-aligned comparison → CSV/JSON/MD + figures.
- `scripts/slurm/compare_quality_cpu.sbatch` — reusable CPU job.
Ran on a compute node via `srun` (~6 min, `-c 8 --mem 32G`). Outputs in
`processed/eog_ecg_clean/qc_vs_derivatives/`: `manifest_qc_summary.json`,
`session_quality_metrics.csv` (153), `paired_similarity_metrics.csv` (144),
`session_alignment.json`, `QC_REPORT.md`, `figures/*.png` (13).

**Manifest QC (153)**: status **148 ok / 5 failed**. Failures = trigger/试次 < 200
(sub-023/ses-01=199, sub-024/ses-02=199, sub-024/ses-03=195, sub-026/ses-01=199,
sub-032/ses-02=199) → shape/label/trigger checks fail for exactly those 5. The other 148:
shape [200,58,1000], y[200], 100/100, 250 Hz, 58 ch, 1000 times, **0 NaN/Inf**. aux cleaning
used 153/153; EOG ICs removed in 106 sessions (189 comps), ECG in 43 (74); ICA n_components
retry fired 10×; **no-aux-clean fallback 0×**. Total 6.15 GiB.

**KEY FINDING — official derivatives session ordering is permuted.** For **22/51 subjects**
the paper `.mat` stores sessions in a different order than BIDS sourcedata (our ses-YY).
Naive same-ses comparison produced absurd std ratios (e.g. sub-030 ses-01 264×, sub-018
ses-01 14×, with reciprocal <0.1 in another session of the same subject). We match each of
our sessions to its true official counterpart within a subject via a (std, max|·|) amplitude
fingerprint (robust to ICA), accepting a permutation only when it beats identity by a margin.
Cross-validated: after matching, those subjects' labels become **exact** and trial-corr jumps
(e.g. sub-001/ses-02 was "102/200 order-diff" under same-ses → it's actually our-ses-02 ↔
official-ses-03, exact + corr 0.92). This reinterprets the earlier "trial-order differences"
as session swaps. **Our (X,y) are correct** (sourcedata + evt.bdf; per-subject session
amplitude *set* matches official). Permuted subjects: 001,004,006,007,008,010,013,017,018,
023,025,027,030,036,038,039,042,044,045,047,049,051.

**Comparison verdict (after alignment) = PASS**:
- std ratio (ours/official) median **0.976** (111/153 ours lower → artifact removal), RMS
  ratio 0.976; μ/α bandpower ratio **0.898**, β ratio **0.941**.
- paired (144 exact-label sessions) trial-wise Pearson median **0.954**, rel-RMSE 0.251.
- PSD overlays (global + C3/C4/Cz) overlap through μ/β; only 40–50 Hz differs (MNE firwin +
  50 Hz notch vs EEGLAB rolloff — above MI bands, expected).
- MI class separability (|Cohen's d| on log-bandpower) ours≈official at C3/C4 μ/β
  (e.g. C3 μ 0.138 vs 0.150; C4 β 0.142 vs 0.144) → discriminative MI info preserved.
- 22 "attention" sessions = the 5 failed + a few heavy-clean low-ratio sessions (std ratio
  0.35–0.42, genuine artifact removal) + genuinely high-amplitude recordings (>500 µV trials,
  also high in official) — none are data bugs.

**Recommendation**: **proceed to 41/10 split + SHUTrialDataset** using the **148 ok** sessions
(`SHUTrialDataset.from_manifest(..., statuses=('ok',))` already filters). First decide the 5
failed sessions: exclude, or re-extract triggers from raw (their subjects have other ok
sessions, so subject-level 41/10 split is unaffected). **Still deliberately NOT done**:
training, 41/10 split, CAP-EEGNet, any GPU job — awaiting user go-ahead.

---

## 2026-06-04 — Small-batch validation: sub-001/002/003 × 3 = 9 sessions (9/9 ok)

**Run**: `srun ... python scripts/preprocess_all.py --subjects 1,2,3 --tag dryrun` (~313 s,
compute node). Outputs to the external eog_ecg_clean tree; wrote
`processed_manifest.dryrun.csv` + `preprocess_summary.dryrun.csv` (canonical names reserved
for the full run). **9/9 ok**, every session shape [200,58,1000], float32/int64, 100/100
labels, 58 ch, 1000 times, 250 Hz, no NaN/Inf.

**New fail policy (req): `evaluate_failure_reasons()`** in `eog_ecg_clean.py`, applied after
the `.mat` cross-check in `pipeline.py`. A session is FAILED iff: shape≠[200,58,1000], label
count≠100/100, trigger count≠200, NaN/Inf in X, or (when `.mat` exists)
`labels_multiset_match=False`. `labels_match_mat=False` (order-only diff) is NOT a failure.
Added `labels_multiset_match` + `n_labels_agree` columns to the processed manifest + summary.

**Per-session results (excluded ICs / match_mat / multiset)**
- sub-001: ses-01 [1,4,10] T/T; ses-02 [1] F/T (order diff, 102/200); ses-03 [1,2] —/— (no .mat).
- sub-002: ses-01/02/03 all [] (none) T/T. Top EOG |corr| only ~0.10–0.14 → genuinely little
  IC-level ocular contamination (NOT a near-miss); sub-002 ses-01 ECG flat/invalid. Removing
  nothing is the correct conservative outcome.
- sub-003: ses-01 [1] T/T; ses-02 **[0,1,3,9,12,15,16]** T/T — 4 EOG (|corr| 0.51–0.75) + **3
  ECG (|corr| 0.56–0.77)**: first session with real cardiac ICs → ECG path validated; ses-03
  [3] T/T (see below).

**ECG**: detected in every session, but only sub-003/ses-02 had ECG-correlated ICs (≥0.5).
Elsewhere max |corr|~0.002–0.03 → no cardiac IC, not removed. Confirms ECG cleaning fires
only when there's genuine contamination.

**Robustness fix — ICA n_components retry.** sub-003/ses-03 is an EXTREME-amplitude session
(std ≈ 660 µV, ~60× normal; the paper `.mat` agrees at 660 → real data). `n_components=0.99`
collapsed to 1 PCA comp → MNE `ICA.fit` raised. Added a retry: on fit failure, refit with a
fixed int `aux_cleaning.ica.n_components_fallback` (=15). After the fix, ses-03 fits with 15
comps and removes EOG IC [3] (|corr| 0.653); the warning self-documents the retry. If the
retry also fails, the existing no-aux-clean fallback still applies (recorded, never crash).

**Disk**: each npz ≈ 41 MiB (≈43 MB; float EEG barely compresses), 9 sessions ≈ 392 MB →
full 153 ≈ ~6.5–7 GB. 38 files = 9×{npz,meta.json,preprocess_report.json,manifest_row.json}
+ 2 CSVs.

**Open questions**: (1) sub-001/ses-02 (and likely others) store trials in a different order
than the paper `.mat` — multiset matches, our (X,y) are evt.bdf-consistent; treat exact
`labels_match_mat` as non-authoritative. (2) Some subjects (sub-002) yield no ocular ICs at
threshold 0.5 — confirmed genuine here; revisit only if many subjects look under-cleaned.

**Deliberately NOT done**: full 51×3 run, training, 41/10 split, GPU. Awaiting user go-ahead.

---

## 2026-06-04 — EOG/ECG-clean preprocessing, .npz output, dry-run (sub-001)

**What changed (files)**
- `configs/paths.yaml`: renamed processed keys to `*_root`; `eog_ecg_clean_root` +
  `manifests.processed_manifest` now point to the external
  `/share/workspace2/moto_imagination/WBCIC_SHU/processed/eog_ecg_clean/...`.
  `paper_style_root` kept (legacy sanity-check only, not the formal output).
- `configs/preprocess.yaml`: `mode: eog_ecg_clean`; `output.{format=npz, npz_compress,
  save_debug_npy=false, write_manifest_row}`; `aux_cleaning.{enabled, method=ica,
  fallback, validity, ica{...}}`.
- NEW `src/preprocessing/eog_ecg_clean.py`: aux-channel validity check + MNE ICA cleaning
  (fit EEG on 1 Hz-highpass copy; detect via EOG/ECG; exclude; apply) + paper-style 2nd
  half + 11-point QC report. ICA/detection failures degrade to no-aux-clean (recorded,
  never crash).
- NEW `src/preprocessing/pipeline.py`: per-session process→save(.npz/meta/report/
  manifest_row)→build manifest row; richer `.mat` cross-check. Shared by both scripts.
- `src/utils/paths.py`: `*_root` fields, `session_npz_path()`, `assert_safe_output_dir()`
  (refuses sourcedata/derivatives/raw root), `ensure_writable_dir()` (errors if unwritable).
- `src/utils/io.py`: `save_session_npz()`/`load_session_npz()`/`save_debug_npy()`.
- `src/data/manifest.py`: `PROCESSED_MANIFEST_FIELDS` + `build_processed_manifest_row()`
  (records `npz_path`, NOT X/y paths).
- `scripts/preprocess_raw.py`: dispatch on `mode`; eog_ecg_clean → .npz + report + safety.
- `scripts/preprocess_all.py`: implemented (manifest-driven; `--subjects/--sessions/
  --limit/--tag`; per-session try/except → status+error_message; writes
  `processed_manifest.csv` + `preprocess_summary.csv` beside the npz tree).
- `src/data/shu_dataset.py`: implemented — reads X/y from `npz_path` (LRU-cached, lazy
  index from y), `from_manifest()` / `from_npz_paths()`.

**.npz keys**: `X` [200,58,1000] float32 (µV), `y` [200] int64 (0/1), `subject_id`,
`session_id`, `sfreq` (=250), `channel_names` (58).

**Dry-run (sub-001, `srun` compute node, ~36 s/session, mode=eog_ecg_clean)** — 3/3 ok:
- All sessions: shape [200,58,1000], dtype float32/int64, 100/100 labels, no NaN/Inf,
  58 ch, 1000 times, 250 Hz → every QC `all_passed=true`.
- Aux cleaning REAL (correlation-based IC detection, |corr|>0.5):
  - ses-01: valid EOG=HEOR/HEOL/VEOU (VEOL flat→invalid), ECG valid; excluded EOG ICs
    [1,4,10] (corr 0.66/0.65/0.61; gap to next 0.46). std 11.28→9.99.
  - ses-02: all 4 EOG + ECG valid; excluded EOG IC [1].
  - ses-03: all valid; excluded EOG ICs [1,2] (0.69/0.61).
  - ECG: detected in every session but max IC corr ~0.02–0.03 (both `ctps` and
    `correlation` agree) → no cardiac IC removed. Plausibly little cardiac contamination
    in this montage; recorded, not forced.
- Disk: ~42–68 MB/session (compressed npz), 150 MB for 3 → full 153 ≈ ~7–8 GB.

**KEY FINDING / open question — ses-02 `.mat` label order**
- ses-01 labels match the paper `.mat` 200/200 (exact). ses-02: exact match only 102/200
  (≈chance) BUT multiset identical (100/100) and triggers are monotonic/evenly-spaced
  (same structure as ses-01). ⇒ the paper `.mat` stores ses-02 trials in a DIFFERENT
  order than acquisition; our chronological evt.bdf order pairs each `(X_i, y_i)` from the
  SAME trigger (self-consistent, correct). Cross-check now reports `labels_match` (exact)
  + `labels_multiset_match` + `n_labels_agree`. Treat `labels_match_mat` (exact) as
  non-authoritative for ordering; multiset is the meaningful check.

**Decisions**
- IC detection uses absolute-correlation (`measure='correlation'`, thr 0.5) instead of the
  MNE z-score default (3.0), which detected nothing on ses-01/02; ECG default `correlation`
  (ctps found nothing). All thresholds live in `configs/preprocess.yaml`.
- npz is `savez_compressed`; `X.npy/y.npy` only if `output.save_debug_npy=true`.

**Deliberately NOT done (await user go-ahead)**: full 51×3 `preprocess_all` (only
`--tag dryrun` files written; canonical `processed_manifest.csv` not created), CAP-EEGNet,
training, 41/10 split, any GPU job.

**Next step (after user confirms)**: run full `preprocess_all` on a compute node
(`srun`/sbatch), write canonical `processed_manifest.csv` + `preprocess_summary.csv`,
review failures/`labels_match_mat`, then 41/10 splits + CAP-EEGNet.

---

## 2026-06-04 — Paused at framework + single-session sanity check

**State**
- Framework, rules (7), configs (6), docs (11), and src/scripts skeletons complete and
  committed; working tree clean.
- Raw manifest built: `manifests/shu_2c_raw_manifest.csv` (51 subjects, 153 sessions,
  0 missing BDFs).
- Single-session sanity check RETAINED: `outputs/processed_paper_style/sub-001/ses-01/`
  = {X.npy [200,58,1000], y.npy, meta.json, manifest_row.json}; validated vs the paper
  `.mat` (labels_match=True, corr ~0.994, std 11.28 vs 11.26).
- Slurm queue checked: no jobs running/queued (today's `mi_pp_test` srun checks all COMPLETED).

**Deliberately NOT done (await user go-ahead)**
- Full 51×3 `preprocess_all` (Slurm CPU job).
- Any Slurm submission / GPU job.
- Model training code, CAP-EEGNet implementation, 41/10 split.

**Next step (after user confirms)**: run full preprocessing via the Slurm CPU job,
write `manifests/shu_2c_processed_manifest.csv`, then 41/10 splits + CAP-EEGNet.

---

## 2026-06-04 — Architecture refactor: external paths + manifests + CAP-EEGNet

**Why**: new direction — drop the baseline-first plan; main model is **CAP-EEGNet**;
raw data live OUTSIDE the repo and all paths come from `configs/paths.yaml`; MATLAB is
reference-only; everything Python/PyTorch. Current stage stays data/paths/preprocessing.

**Done**
- Dirs: added `manifests/`, `splits/`, `outputs/processed_*`; removed `data/`, `notebooks/`.
- Rules renamed/rewritten: `00-project-context` (new content), `10-data-paths` (NEW),
  `20-preprocessing`, `30-model-experiments`, `40-online-learning`, `50-server-slurm`
  (+ updated `90-agent-behavior`). Deleted old 10/20/30/40.
- `src/utils/paths.py` refactored: no hardcoded dataset root; `load_paths()` reads
  `configs/paths.yaml` (env `SHU_2C_ROOT` overrides), validates, returns a `Paths` object.
- `configs/`: NEW `paths.yaml`; `preprocess.yaml` slimmed to params only; renamed
  cross_subject→`train_cross_subject.yaml`, online→`online_adaptation.yaml` (CAP-EEGNet).
- `scripts/build_manifest.py` + `src/data/manifest.py`: scan external raw root →
  `manifests/shu_2c_raw_manifest.csv`. Verified: **51 subjects, 153 sessions, 0 missing**.
- `check_raw_bdf.py` + `preprocess_raw.py` now read `configs/paths.yaml`; preprocess
  writes to the configured processed dir, writes `manifest_row.json`, and (optional)
  cross-checks labels vs the paper `.mat`.
- Skeletons (no complex code yet): `src/models/cap_eegnet.py` (encoder+adapter+
  prototype+confidence+cls), `src/data/splits.py`, `src/data/shu_dataset.py`.
- Re-validated end-to-end on a compute node (srun): sub-001/ses-01 →
  `outputs/processed_paper_style/...`, shape [200,58,1000], **mat labels_match=True**,
  std 11.28 vs 11.26.

**Decision**: `configs/paths.yaml` ships with the verified real raw root
(`/share/workspace2/moto_imagination/WBCIC_SHU`) since it's known; loader still
validates + supports `SHU_2C_ROOT` override. Change the YAML if data moves.

**Next step**: when ready, implement `src/data` (dataset + subject-wise split),
then CAP-EEGNet encoder. Still NO full 51×3 preprocessing / GPU jobs in this stage.

---

## 2026-06-04 — Task 2: raw preprocessing implemented & validated (sub-001/ses-01)

**Done**
- Cracked the evt.bdf event parsing (was the #1 blocker). The 200 MI triggers live
  in the `BDF Annotations` TAL channel of the Neuracle BDF+C file; MNE only surfaces
  the block markers {7,8}. Wrote `src/preprocessing/neuracle_events.py` to parse the
  BDF header + TAL bytes directly -> recovers 100x'1'(left) + 100x'2'(right), ~8s apart.
- Implemented `src/preprocessing/shu_preprocess.py` (paper-style), faithful to the
  authors' `code/pre-processed/preprocessed.m`:
  drop {ECG,HEOR,HEOL,VEOU,VEOL} -> reref Pz & drop Pz (58 EEG) -> 0.5-40 bandpass
  -> 50 notch -> epoch [0,4)s (baseline = whole-epoch demean) -> resample 250 -> [200,58,1000].
- Wired `scripts/preprocess_raw.py`; ran on a COMPUTE NODE via `srun` (not login node).

**Validated vs paper .mat (derivatives/)**
- Shape [200,58,1000], labels match **exactly** (element-wise, all 200 trials).
- Signal correlation 0.988-1.000 (mean 0.994). After fixing a unit bug, scale matches:
  our std 11.283 vs paper 11.263 (ratio 1.0017), RMSE 0.876 uV (~7.8% of std). The
  residual is expected from EEGLAB-vs-MNE filter implementation differences.

**Decisions / gotchas**
- UNIT QUIRK: BDF physical dim is the garbled `?V` (meant µV), so MNE does NOT apply
  µV->V scaling; `get_data()` already returns µV-magnitude values. We store as-is
  (µV) and do NOT multiply by 1e6 (doing so was a bug, fixed).
- reref target Pz = EEGLAB channel index 43 (confirmed by counting the EEG montage).
- Epoch window [0,4)s at 1000Hz (4000 samples) then resample -> 1000 samples.

**Next step**
- Generalize to all 51x3 via `scripts/preprocess_all.py` + `scripts/slurm/preprocess_cpu.sbatch`
  (collect per-session status into outputs/preprocess_summary.csv; don't silently skip
  failures). Then start the EEGNet baseline.

---

## 2026-06-04 — Project scaffold created

**Done**
- Created project skeleton at `/share/home/yuan/SYX/eeg-mi-online/`:
  `.cursor/rules/` (6 rules), `docs/` (8 docs + references), `configs/`,
  `scripts/` + `scripts/slurm/`, `src/` (7 packages), `data/`, `outputs/`,
  `logs/`, `checkpoints/`, `notebooks/`.
- Wrote the 6 Cursor rules, the docs (PROJECT_BRIEF, DATASET_SHU,
  PREPROCESSING_SPEC, EXPERIMENT_PROTOCOL, MODEL_PLAN, SERVER_RUNBOOK, ENVIRONMENT),
  AGENTS.md, this file, `.gitignore`, `requirements.txt`.
- Copied the senior's chat record to `docs/references/ChatGPT-EEG-MI-pretraining.md`.
- Implemented `scripts/check_raw_bdf.py` (raw BDF inspector, Task 1).
- Added documented stubs for the other scripts + `src/` modules.
- Wrote Slurm sbatch templates in `scripts/slurm/` (adapted from `run_test.sh`).
- Initialized git and made the first commit.

**Verified facts (from inspecting the real server/dataset)**
- Dataset root: `/share/workspace2/moto_imagination/WBCIC_SHU` (BIDS, READ-ONLY).
- 2C: **51** subject folders on disk; `participants_2C.tsv` lists 52; README says 53.
  -> Decision: always enumerate subjects from disk, never hardcode the count.
- Raw: 1000 Hz, **64 ch**. `check_raw_bdf.py` on sub-001/ses-01 VERIFIED the real
  layout = **59 EEG + 1 ECG (`ECG`) + 4 EOG (`HEOR/HEOL/VEOU/VEOL`)**. The
  `task-motorimagery_eeg.json` count "1 EOG, 4 ECG" is SWAPPED vs the actual names
  -> trust the names. (The generic plan's "ch60=ECG, ch61-64=EOG" was actually
  closer to reality than the JSON.)
- Events: 1=left, 2=right, 3=foot(3C only). 2C uses {1,2} -> internal {0,1}.
- Target processed shape `[200, 58, 1000]` (58 = 59 EEG minus Pz; 1000 = 4 s @ 250 Hz).
- Paper `.mat` confirmed: `data [58,1000,200]` float32, `labels [1,200]` in {1,2}.
- Env `mi_torch`: py3.10.18, torch 2.6.0, mne 1.10.0, numpy 2.2.5, scipy 1.15.3,
  sklearn 1.7.1, pandas 2.2.3, h5py 3.14.0, einops 0.8.1. No braindecode.
- Slurm: `gpu2node`(default)/`gpu3node`, each `gpu:8`/128 CPU/~773 GB; modules
  `cuda/11.8`, `anaconda3`.

**Decisions**
- Two preprocessing variants: paper-style (first, no ICA) and EOG/ECG-clean (later).
- First model = EEGNet encoder + classification head + prototype + confidence head
  + adapter (the chat record's "minimal version"). Baselines (EEGNet/DeepConvNet/
  FBCNet) come before that, to validate preprocessing.
- Git identity for commits: see commit log; tell the maintainer if it needs changing.

**check_raw_bdf.py first run (sub-001/ses-01) — verified**
- sfreq=1000 Hz, 64 ch, duration=2250 s (~37.5 min; ~11.25 s/trial for 200 trials).
- Channels: 59 EEG + 1 ECG(`ECG`) + 4 EOG(`HEOR/HEOL/VEOU/VEOL`). `other`=[] after
  improving the classifier to recognize H/V-EOG names.
- Aux validity: first 10 s of `ECG` and `VEOL` were all-zero (flat); `HEOR/HEOL/VEOU`
  active. -> must validate aux over the full recording before use (variant 2).
- Report saved at `outputs/raw_check/sub-001_ses-01_raw_check.json`.

**Open questions / TODO before scaling up**
- ⚠️ `mi_torch` torch is **CPU-only** (`torch.version.cuda is None`). Must install a
  cu118-matched torch (or make a `mi_torch_cu118` env) before real GPU training.
  (RESOLVED later: `mi_torch_cu118` env created, torch 2.7.1+cu118.)
- ⚠️ **Event triggers**: `evt.bdf` annotations via MNE gave only `{"7":1,"8":1}`
  (channel "Empty Event Data"), NOT 200 MI markers. (RESOLVED: TAL parser in
  `neuracle_events.py`.)
- Confirm cue timing for the 4 s epoch window + baseline interval (paper detail TBD).
- Check whether the flat ECG/VEOL is session-specific or dataset-wide.

**Next step**
- Run `scripts/check_raw_bdf.py` on `sub-001/ses-01` (login-node-safe: reads one
  file, prints + dumps JSON), read the report, then implement `preprocess_raw.py`
  for a single session and confirm the `[200, 58, 1000]` shape.
