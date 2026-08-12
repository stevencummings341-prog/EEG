---
title: "File Catalog"
tags:
  - "#modality/eeg"
created: "2026-06-10"
updated: "2026-08-04"
status: "active"
---

# File Catalog

新增 source/config/doc/结果文件时更新本表。

## 根目录

| 路径 | 作用 |
|:---|:---|
| `AGENTS.md` | 唯一权威灵魂记忆。 |
| `CLAUDE.md` | 兼容入口，指向 AGENTS.md（供 Claude Code 等原生读取）。 |
| `README.md` | 人类入口与结构总览。 |
| `FOUNDATION_E2E_ROUTE_PLAN.md` | **当前主线路线（2026-08-04）**：端到端基础模型（5 个 S4/DINO-DualCD）× 跨被试 × 双数据集分开训练。含融合成果映射、运行命令、checkpoint/断点续跑契约、协议默认值与 7 个待学长确认问题、7 条硬约束。 |
| `PHASE3_ROUTE_PLAN.md` | 上一条主线 Phase 3 路线 v2.1（Oracle 先裁决；学长已批 A0/A1）；**paused，不是废弃**。 |
| `proposal.md` | 项目提案。 |
| `progress.md` | 进度日记（PROGRESS 角色，逐条追加）。 |
| `experiment_log.md` | 实验日志速查。 |
| `results.md` | 结果速查表。 |

## 0_docs

| 路径 | 作用 |
|:---|:---|
| `0_docs/ARCHITECTURE.md` | 结构 + 代码分层 + 文件职责。 |
| `0_docs/STATUS.md` | 进度 + 运行就绪 + SHU 就绪 + 下一步 + 清理策略。 |
| `0_docs/FILE_CATALOG.md` | 本文件，文件索引。 |
| `0_docs/operation_log.md` | 文件系统操作日志。 |

## 阶段结果

| 路径 | 作用 |
|:---|:---|
| `1_session_drift/{wbci_shu,shu}/{report,tables,figures}/` | Phase 0 漂移诊断结果（数据集并列）。 |
| `2_baseline/{wbci_shu,shu}/no_alignment_baseline/{report,tables,figures}/` | Phase 1 baseline + Phase 2a multi-source（两数据集 done；含 `report/AI_ANALYSIS.md`）。 |
| `2_baseline/{wbci_shu,shu}/alignment_baseline/{report,tables,figures}/` | Phase 2b alignment（两数据集 done；含 `report/AI_ANALYSIS.md`）。 |
| `4_experiments/{wbci_shu,shu}/prototype_drift/{report,tables,figures}/` | Phase 2c prototype drift（两数据集 done；含 `report/AI_ANALYSIS.md`）。 |
| `PHASE3_ROUTE_PLAN.md`（根目录） | **Phase 3 完整路线计划 v2.1（已批准 A0/A1）**：Oracle 先裁决 + 7 条硬约束。 |
| `3_online_adaptation/PHASE3_TTA_DESIGN.md` | **Phase 3 技术总纲**：T3A 方法/公式/矩阵/诚实警示 + 实现路线 v2。 |
| `3_online_adaptation/PRETRAINED_MODEL_INTEGRATION_CONTRACT.md` | **预训练模型接入权威契约**（交付物/能力矩阵/步骤/preflight）。 |
| `3_online_adaptation/` | **设计文档区**；正式实验结果不放这里。 |
| `4_experiments/{wbci_shu,shu}/tta/` | **Phase 3 TTA 正式结果区**（smoke / full A0 replay / oracle_diagnostic / method_catalog / reports）。 |
| `4_experiments/CROSS_SUBJECT_PROTOCOL_MEMO.md` | **发学长的跨被试协议讨论备忘**：进度一句话、4 个文献关键发现（含 EDAPT 对标数字与 SHU 地板效应）、3 个候选方案的算力对比、7 个待拍板问题。status=`pending_advisor_confirmation`。 |
| `4_experiments/{wbci_shu,shu}/foundation_cross_subject/` | **（未来）端到端主线结果区**；协议确认并跑完后才建 `report/tables/figures`。 |
| `5_papers/{wbci_shu,shu}/` | 论文材料。 |
| 每层目录的 `README.md` | 数据集层/实验层/叶子层说明，由 `scripts/scaffold_readmes.py` 生成。 |

## 代码

| 路径 | 作用 |
|:---|:---|
| `code/run.py` | 统一入口：dry-run / 训练 / `--summarize`，进程内调度。端到端主线新增 CLI：`--split-protocol` / `--folds-subset` / `--monitor` / `--no-resume`。 |
| `code/runners.py` | Phase 0/1/2a/2b/2c/**phase3_tta**/**foundation_cross_subject** 的进程内 runner。 |
| `code/models/eeg_foundation/` | **端到端主线 5 模型包**（学长 `models_eeg_foundation/` 移植）：`s4_layers.py`（S4 HiPPO+FFT）、`pooling.py`（flatten/attention/temporal-bin）、`encoders.py`（ShallowNet stem + S4/Transformer）、`losses.py`（DINO/iBOT/DKoleo/PrototypeBank/OrthogonalMask/双扰动）、`models.py`（5 变体 + MultiViewGenerator）、`adapter.py`（**新增**：项目契约 + DualCD 训练钩子 + per-trial 归一化）。 |
| `code/models/eeg_foundation/README.md` | 5 模型对照表、两套「方言」的桥接方式、**移植偏差逐条记录（§4）**、MI 适配（mu/beta 视图、4s 时间分箱）。 |
| `code/training/e2e_trainer.py` | 端到端训练器：非 CE 损失钩子、**每 cell 只存 `best.pt`/`last.pt`**、断点续跑（optimizer/scheduler/RNG/history + 原子写入 + `run_signature` 配置漂移守卫）、cosine/梯度裁剪/可选 AMP。 |
| `code/experiments/cross_subject_protocols.py` | 跨被试协议：被试级 `loso`/`kfold_subject`/`holdout`；`val_mode`=`subjects`/`trials`/`sessions`（后者对齐 DSGNet/SHUv5：train ses-01+02、val ses-03）；per-trial 归一化、泄漏断言、`best`+`last` 双评测、`cell_signature`。 |
| `code/configs/experiments/foundation_cross_subject_wbci_3c.yaml` | WBCIC-SHU **三分类 11 人** LOSO 端到端（`run_id=foundation_3c_loso_paper_v1`，论文对齐 session val）。 |
| `inbox/papers/dsgnet_jbhi2026_FullText.pdf` | Lou et al. IEEE JBHI 2026（DSGNet）全文 PDF；对标 Acc/F1/Kappa。 |
| `inbox/papers/dsgnet_jbhi2026_extracted.txt` | 同上 PDF 的纯文本摘录（协议与 Table II）。 |
| `4_experiments/wbci_shu/foundation_cross_subject/` | 3C 端到端可读区：对标锚点 + README；重结果在 `outputs/.../foundation_3c_loso_paper_v1/`。 |
| `code/configs/models/{s4erp,dualcd_s4_pos,dualcd_s4_timepatch,dualcd_s4_flatten,dualcd_transformer}.yaml` | 5 个端到端模型结构超参（含 MI 频带视图与 4s 时间分箱）。 |
| `code/models/atcnet/` | **ATCNet（Altaheri 2023）已发表基线**，对标 DSGNet 论文 Table II。`_official_keras/`=官方仓库 [Altaheri/EEG-ATCNet](https://github.com/Altaheri/EEG-ATCNet) 原样（Keras，不执行，供逐行核对）、`atcnet_torch.py`=官方 `ATCNet_` 的 1:1 PyTorch 移植（BCI-IV-2a 维度参数量 113,732 与官方 README 一致）、`adapter.py`=项目 dict 契约 + 官方 L2 penalty 钩子、`README.md`=出处/保真证据/偏差清单。 |
| `code/configs/models/atcnet.yaml` | ATCNet 结构超参（**官方 `ATCNet_` 默认值，不调参**）。 |
| `code/models/paper_baselines/` | **DSGNet 论文 Table II 的已发表 baseline（只用各自作者官方仓库）**。`_official/`=官方源码原样（EEGNet/EEGNeX 的 Keras 仅供核对不执行；`EEGDeformer.py` 是官方 PyTorch，直接执行）、`eegnet_official_torch.py` / `eegnex_official_torch.py`=1:1 移植、`keras_compat.py`=TF same padding / max_norm / glorot 初始化、`adapter.py`=项目 dict 契约、`README.md`=出处/保真证据/偏差/**排除清单及理由**（EEG-Inception、MDGEEG、EEG-DG、DSGNet 无完整官方码）。 |
| `code/configs/models/{eegnet_official,eegnex,eeg_deformer}.yaml` | 三个 baseline 的结构超参（**全部上游作者默认值**）。 |
| `code/configs/experiments/paper_baseline_3c_821.yaml` | **7 模型统一对比 run**：WBCIC 3C、8:2:1 跨被试、论文 recipe（Adam 1e-4/batch 128/500ep）+ 早停 patience 100 + 三曲线。 |
| `scripts/slurm/submit_paper_baseline_821.sh` | 提交/续跑 `paper_baseline_3c_821_v1`：每未完成 fold 一个 job + `--models` 只跑缺的模型（QOS 单 job 上限 48h，无法加长）。 |
| `4_experiments/wbci_shu/foundation_cross_subject/HANDOFF_821_RUN.md` | **821 run 交接文档**：实验设定、当前 50/77 进度与阶段数字、参数/代码位置、换账号要拷哪些产物（数据 2.0G + outputs 14M 必需，checkpoints 8.8G 可精简）、续跑与进度查询命令、8 条坑。 |
| `scripts/plot_three_curves.py` | 从 `result.json` 的 `history` 画 per-epoch train/val/test 三曲线（per-cell + 跨折均值）。 |
| `scripts/summarize_cross_subject.py` | 跨被试 run 汇总：`tables/{per_cell,per_model,vs_paper}.csv` + `REPORT_TABLE.md`（内嵌论文 Table II SHUv5 数字作对照）。 |
| `code/configs/experiments/{foundation_cross_subject,shu_foundation_cross_subject}.yaml` | **端到端主线双数据集配置**（WBCIC 58ch / SHU 32ch，分开跑；协议段标注 pending_advisor_confirmation）。 |
| `tests/foundation/test_eeg_foundation_contract.py` | 5 模型 × 双通道数的 forward 契约、loss 可反传、teacher EMA/prototype 更新、归一化 fit-free、参数量对齐学长表格（19 passed, CPU）。 |
| `tests/foundation/test_cross_subject_protocol.py` | 划分互斥/决定性、只写 best+last、完成 cell 跳过、中断 cell 续跑、换了划分/超参后拒绝续跑、**测试被试不进 train/val**、通道不匹配报错（13 passed, CPU）。 |
| `code/tta/` | **Phase 3 model-agnostic TTA backend**（adapters/feature_sources/methods/oracle/eval/report）。 |
| `code/tta/README.md` | TTA 包概览 + 契约指针。 |
| `code/tta/method_catalog.yaml` | 方法候选清单（多数只登记不实现）。 |
| `code/experiments/session_tta.py` | Phase 3 编排：smoke / opt-in `full_a0_replay` / SHU smoke。 |
| `code/methods/t3a.py` | 薄 re-export → `code.tta.methods.MinimalT3AMethod`。 |
| `code/experiments/prototype_drift.py` | Phase 2c：source-only 训练 + 冻结 embedding 提取 + prototype + 漂移指标 + 泄漏断言。 |
| `code/experiments/prototype_drift_summarize.py` | Phase 2c 汇总：合并 per-run CSV → tables/figures/report/run_status。 |
| `code/configs/experiments/phase2c_prototype_drift.yaml` | Phase 2c 实验配置（WBCIC）。 |
| `code/configs/experiments/shu_phase{0_drift_diagnostic,1_baseline,2b_alignment,2c_prototype_drift}.yaml` | SHU 4 个实验配置（32ch/5 session，数据集作用域输出）。 |
| `code/configs/experiments/{phase3_tta,shu_phase3_tta}.yaml` | **Phase 3 TTA 配置**（默认 smoke 安全开关）。 |
| `code/configs/experiments/phase3_tta_full_a0.yaml` | **Opt-in** WBCIC full A0 replay（非默认）。 |
| `tests/tta/` | TTA 行为测试（含 mock live-inference fixtures；CPU）。 |
| `code/preprocessing/shu_mat.py` | SHU per-session `.mat` → 标准化 `.npz` 的核心加载/校验。 |
| `scripts/preprocess_shu.py` | SHU 全量预处理：`.mat` → npz + `processed_manifest.csv`。 |
| `scripts/build_drift_report.py` | session/dataset 无关的 drift per-pair/per-subject 表+图+报告构建器（泛化 legacy WBCIC-only 版）。 |
| `scripts/make_baseline_cross_all.py` | Phase 2b baseline schema 适配器：Phase 1 cross（accuracy/train_session）→ alignment 口径 `results_cross_session_all.csv`（acc/train_sessions/training_scope），数据集无关。 |
| `scripts/scaffold_readmes.py` | 双数据集结果树各层 README 生成器。 |
| `code/configs/paths.example.yaml` / `paths.yaml` | 路径模板与仓库占位；真实路径在 `paths.local.yaml`。 |
| `code/configs/paths.local.yaml` | **本机**路径（gitignore，不推送）。 |
| `code/configs/datasets/{shu,wbci_shu}.example.yaml` | 数据集路径模板。 |
| `code/configs/datasets/{shu,wbci_shu}.local.yaml` | **本机**数据集路径（gitignore）。 |
| `scripts/slurm/_common.sh` | Slurm 共用：自动探测 PROJECT_ROOT + conda activate。 |
| `scripts/slurm/shu_gpu.sbatch` | 通用 GPU 训练 Slurm 脚本（config + passthrough，mi_torch_cu118）。 |
| `scripts/slurm/shu_cpu.sbatch` | SHU 通用 CPU Slurm 脚本（Phase 0 drift / `--summarize`）。 |
| `checkpoints/README.md` | checkpoint 命名规范（dataset/run_id/method/model/任务前缀/sub/session/seed）。 |
| `scripts/slurm/train_prototype_drift_gpu.sbatch` | Phase 2c 单 (model,seed) GPU 训练 job。 |
| `scripts/slurm/summarize_prototype_drift_cpu.sbatch` | Phase 2c CPU 汇总 job（afterany 依赖）。 |
| `scripts/slurm/submit_prototype_drift_full.sh` | 提交 15 GPU + 1 summarizer 全量任务并记录 job ids。 |
| `code/summaries/session.py` | Phase 1 baseline 汇总（表/图/原始报告）。 |
| `code/summaries/multisource.py` | Phase 2a multi-source 汇总。 |
| `code/summaries/alignment.py` | Phase 2b alignment 汇总。 |
| `code/summaries/canonical.py` | 按 9 段结构从汇总 CSV 生成 canonical `REPORT.md`。 |
| `code/summaries/summarize.py` | `--summarize` 调度器：原始汇总 + canonical。 |
| `code/configs/datasets/*.yaml` | 数据集配置（wbci_shu / shu）。 |
| `code/configs/models/*.yaml` | 模型超参配置。 |
| `code/configs/experiments/*.yaml` | 阶段实验配置。 |
| `code/datasets/` | 数据集适配器与 split/通道映射。 |
| `code/models/` | 模型与 registry。 |
| `code/methods/` | 对齐与适应方法。 |
| `code/experiments/` | 漂移/baseline/multi-source/alignment/跨被试 协议 + 指标。 |
| `code/training/` | 通用 trainer（`trainer.py` 冻结）+ 端到端 trainer（`e2e_trainer.py`）。 |
| `code/preprocessing/` | 预处理逻辑。 |
| `code/utils/` | config/io/logging/paths/seed。 |

## inbox

| 路径 | 作用 |
|:---|:---|
| `inbox/README.md` | inbox 职责与归档规则。 |
| `inbox/cross_subject_protocol_research.md` | 跨被试（subject-independent）实验协议文献调研：SHU 2022 + WBCIC-SHU 2025 的论文清单、协议细节、可参考准确率区间、数据集陷阱，以及 3 个候选协议方案（LOSO / 5-fold subject-grouped / 固定划分）与算力代价对比。**端到端主线的协议依据**；结论摘要见 `4_experiments/CROSS_SUBJECT_PROTOCOL_MEMO.md`。 |

## backup

| 路径 | 作用 |
|:---|:---|
| `backup/README.md` | backup 索引与策略。 |
| `backup/COMPLETED_ARTIFACTS_INDEX.md` | 历史结果/权重/日志归档位置。 |
| `backup/root_archive_2026-06-10/` | 清理前的旧代码/文档/产物/权重/日志。 |
| `backup/legacy_snapshot_2026-06-10/` | 更早的轻量快照。 |
