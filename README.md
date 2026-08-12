---
title: "EEG-MI Online 项目入口"
tags:
  - "#pipeline/5_dl_model"
  - "#modality/eeg"
  - "#method/domain_generalization"
created: "2026-06-10"
updated: "2026-08-04"
status: "active"
---

# EEG-MI Online

> 多数据集运动想象 EEG 泛化研究。**当前主线（2026-08-04 学长重新定向）= 端到端基础模型 × 跨被试**：
> 用 5 个 S4 / DINO-DualCD 模型在 WBCIC-SHU 与 SHU 上**分开**做端到端训练 + 跨被试评测。
> 融合已完成可运行；协议参数待学长确认；正式实验未跑。

## 0. 当前定位

**主线：端到端基础模型 × 跨被试**（路线 = 根目录 `FOUNDATION_E2E_ROUTE_PLAN.md`）。学长交付的
`models_eeg_foundation/` 5 个模型已融入框架：`code/models/eeg_foundation/` + 可断点续跑训练器
`code/training/e2e_trainer.py`（每 cell 只存 `best.pt` + `last.pt`）+ 跨被试协议
`code/experiments/cross_subject_protocols.py` + 双数据集 config，`tests/foundation/` 32 passed。
**下一步卡在协议确认**：`4_experiments/CROSS_SUBJECT_PROTOCOL_MEMO.md` 里 7 个问题要发学长拍板
（文献依据 `inbox/cross_subject_protocol_research.md`）。**协议确认前不产出任何被当作结果的数字。**

**已完成（保留只读）**：两数据集 Phase 0/1/2a/2b/2c 全部完成（WBCIC 4320 cells、SHU 7500 cells 全 ok，
AI 分析已写，结果在 `4_experiments/{wbci_shu,shu}/prototype_drift/`）。核心结论：无学习统计对齐不足
（均无方法过 +2pp）；跨 session 掉点主机制为 **within-class scatter 膨胀 / Fisher collapse（非 centroid
collapse）**，cosine 几何优于 euclidean，FBCNet 几何异常。

**Phase 3（跨 session 修复 / TTA）= paused，不是废弃**：`code/tta/`、`4_experiments/*/tta/`、
`PHASE3_ROUTE_PLAN.md` 全部保留，工程上仍是 pretrained-model-ready（WBCIC full A0 complete）。
等端到端 backbone 出来后可以把 Oracle/T3A 接上去。未运行的内容不写成已完成。

## 1. 目录结构

```text
eeg-mi-online/
├── 0_docs/                 # 文档中心（ARCHITECTURE / STATUS / FILE_CATALOG / operation_log）
├── 1_session_drift/{wbci_shu,shu}/   # Phase 0 漂移诊断结果（数据集并列）
├── 2_baseline/{wbci_shu,shu}/        # Phase 1/2a baseline + 2b alignment 结果
├── 3_online_adaptation/{wbci_shu,shu}/  # 在线适应，future
├── 4_experiments/{wbci_shu,shu}/     # Phase 2c+ 新实验入口
├── 5_papers/{wbci_shu,shu}/          # 论文与汇报材料
├── backup/                 # 旧代码/文档/历史产物/权重/日志归档
├── code/                   # 代码框架，唯一人工入口 code/run.py
├── scripts/slurm/          # Slurm 提交脚本（GPU 训练 / CPU 汇总）
├── inbox/                  # 临时交接材料
├── AGENTS.md               # 唯一灵魂记忆（先读）
├── FOUNDATION_E2E_ROUTE_PLAN.md  # 当前主线路线：端到端基础模型 × 跨被试
├── PHASE3_ROUTE_PLAN.md    # 上一条主线（跨 session 修复；paused）
├── proposal.md             # 项目提案
├── progress.md             # 进度日记（PROGRESS 角色）
├── experiment_log.md       # 实验日志速查
└── results.md              # 结果速查
```

## 2. 运行入口

```bash
cd /path/to/this/repo   # 任意克隆路径
python code/run.py --dry-run --config code/configs/experiments/phase1_baseline.yaml
```

`code/run.py` 已直连 `code/` 模块（含 Phase 0/1/2a/2b/2c runner），可直接训练与 `--summarize`。GPU 任务走 Slurm + `mi_torch_cu118`，先 smoke 再 full。本机数据路径写在 `code/configs/*.local.yaml`（见 `SETUP.md`）。

SHU 直接开跑（同一批 runner，仅 config 不同）：

```bash
python code/run.py --config code/configs/experiments/shu_phase1_baseline.yaml --device cuda
python code/run.py --config code/configs/experiments/shu_phase2b_alignment.yaml --device cuda
python code/run.py --config code/configs/experiments/shu_phase2c_prototype_drift.yaml --device cuda
```

SHU 预处理（若需重生成）：`python scripts/preprocess_shu.py`。

**端到端主线（跨被试，当前）**：

```bash
# 极小 CPU smoke（输出隔离到 *_smoke）
python code/run.py --config code/configs/experiments/shu_foundation_cross_subject.yaml \
    --models s4erp --folds 3 --folds-subset 0 --max-subjects 6 --max-epochs 2 \
    --batch-size 32 --num-workers 0 --device cpu \
    --out outputs/experiments/shu/foundation_cross_subject_smoke \
    --ckpt-dir checkpoints/shu/foundation_cross_subject_smoke

# GPU（Slurm）；重复执行同一条命令 = 断点续跑
sbatch -J e2e_wbcic scripts/slurm/shu_gpu.sbatch \
    code/configs/experiments/foundation_cross_subject.yaml
```

CPU 上 `s4erp` 单 epoch 约 12 分钟（1000 点 × 32ch 空间卷积是瓶颈），**正式 smoke 也要上 GPU 节点**。

## 3. 必读文档

| 文档 | 用途 |
|:---|:---|
| `AGENTS.md` | 唯一权威灵魂记忆。 |
| `progress.md` | 进度日记。 |
| `0_docs/ARCHITECTURE.md` | 目录结构 + 代码分层 + 文件职责。 |
| `0_docs/STATUS.md` | 进度、能否跑、SHU 就绪、下一步、清理策略。 |
| `0_docs/FILE_CATALOG.md` | 文件索引。 |

## 4. 数据边界

外部 raw / processed 路径由本机 `code/configs/paths.local.yaml`（及 `datasets/*.local.yaml`）配置，**不进 Git**。仓库内只有 `/CHANGE/ME/...` 占位。raw 只读；唯一允许写外部盘的位置是各数据集的 `processed/` 子树（若你的 local 配置指向共享盘 processed）。

## 5. 上传 GitHub / 换机继续跑

见根目录 [`SETUP.md`](SETUP.md)：克隆、环境、`paths.yaml` / `shu.yaml` 改路径、数据与冒烟命令。大文件（数据、权重、`outputs/`、`logs/`、`checkpoints/`、`backup/`）不进 Git。
