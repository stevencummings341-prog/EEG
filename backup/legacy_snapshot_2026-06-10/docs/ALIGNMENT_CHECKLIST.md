# ALIGNMENT CHECKLIST — chat record → project status

> 轻量对齐清单：把学长聊天记录 (`docs/references/ChatGPT-EEG-MI-pretraining.md`) 的最终方案，
> 逐项映射到当前项目状态。状态图例：✅ 已实现 / 🚧 未实现（接口/flag 已留）/ 📄 已写进文档规则。
> 详细路线见 `docs/ROADMAP.md`。最后更新：2026-06-05。

## A. 最终目标定义
- [x] 📄 最终目标 = **Confidence-aware Online Adaptive Multi-Subagent Pretraining
      Framework for Cross-subject MI EEG Decoding**（写进 ROADMAP / PROJECT_OVERVIEW /
      rule 00 / cap_eegnet.py docstring）。
- [x] 📄 明确「普通 EEGNet 分类只是 Stage 0 baseline，不是最终方法」。

## B. full CAP-EEGNet 模块（论文最终应包含）
- [x] ✅ Main encoder（EEGNet-style）—— `src/models/eegnet.py`，已跑通。
- [ ] 🚧 Neural subagents / neural experts（深度可微，非手工特征）—— flag+stub `NeuralSubagentEncoder`。
- [ ] 🚧 Confidence head（多源，非 softmax max）—— flag+stub `ConfidenceHead` + `predict_confidence()`。
- [ ] 🚧 Prototype memory（global/subject/session 三级）—— flag+stub `PrototypeMemory`。
- [ ] 🚧 Adapter（轻量适配，微调/在线）—— flag+stub `Adapter`。
- [ ] 🚧 Subject/session domain alignment —— flag+stub `DomainAlignmentHead`。
- [ ] 🚧 Online update module（test-then-update）—— flag+stub `OnlineUpdateModule` + `online_update()`。
- [ ] 🚧 Dataset-aware neural router（可选 v2）—— flag+stub `DatasetAwareRouter`。

## C. 数据与划分
- [x] ✅ full eog_ecg_clean 预处理 + QC：148/153 ok，5 failed，QC PASS。
- [x] ✅ 训练入口 = `status==ok` 的 `.npz`；derivatives `.mat` 仅 QC（`SHUTrialDataset` 默认 statuses=('ok',)）。
- [x] ✅ 41/10 **subject-wise** split，**repeated** seeds 2026–2030（每个 split 一个模型 → mean±std）。
- [x] ✅ target 被试需 3 session 全 ok；source 可含 failed session 但只用 ok session；failed → excluded。
- [x] ✅ 无 target 泄漏、无 session/trial-wise（`tests/test_splits.py` 校验通过）。

## D. 置信度（confidence）
- [x] 📄 规则/文档明确：confidence **绝不**等于 softmax 最大值。
- [ ] 🚧 多源置信度实现：predictive entropy / prototype margin / consistency / OOD-calibration。
- [x] ✅ 校准指标已就绪：`src/evaluation/metrics.py`（ECE / NLL / Brier）。
- [ ] 🚧 risk-coverage / confidence-accuracy curve（待 confidence head 后）。

## E. 实验协议
- [x] 📄 Experiment 1（cross-subject zero-shot）输入/输出/禁止事项已写清。
- [x] 📄 Experiment 2（target Session 1 微调；优先 adapter+prototype+calibration）。
- [x] 📄 Experiment 3（Session 2/3 online test-then-update；禁先更新后测同 session）。
- [x] 📄 Experiment 4（ablation：去 confidence/prototype/adapter/online、softmax-only、无阈值、full-backbone…）。
- [ ] 🚧 上述实验的训练/评估代码（`scripts/train_cross_subject.py` 等仍是骨架，NotImplementedError）。

## F. 在线学习（online）
- [x] 📄 test-then-update 铁律写进 rule 40 + EXPERIMENT_PROTOCOL + ROADMAP。
- [x] 📄 默认禁止 online 更新 full backbone；confidence gate / prototype / adapter 为核心。
- [x] ✅ `configs/online_adaptation.yaml` 默认 `backbone:false`，含置信度阈值/EMA/replay/蒸馏等稳定项。
- [ ] 🚧 `src/online/` 仍是空骨架。

## G. 工程纪律
- [x] ✅ 路径全部走 `configs/paths.yaml`，不硬编码。
- [x] ✅ 重活只在计算节点（srun/sbatch），登录节点不跑训练/全量预处理。
- [x] 📄 GPU torch CPU-only 已记录 + 修复方案（专用 cu118 环境，不动共享 `mi_torch`）。
- [ ] ⏸ 正式训练 / GPU 环境安装 / sbatch 训练任务：**待用户确认后再做**。

## H. 当前阻塞 / 下一步
1. 🚧 GPU torch 仍 CPU-only（`torch.version.cuda=None`）。修复方案见 `docs/ENVIRONMENT.md`，待确认。
2. 🚧 实现 Stage 2 组件（confidence/prototype/adapter）→ full CAP-EEGNet。
3. ⏸ 正式 Experiment 1 训练（需先解决 GPU + 用户确认）。
