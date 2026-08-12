---
title: "端到端基础模型路线计划 v1（跨被试；融合已完成，协议待学长确认）"
tags:
  - "#pipeline/5_dl_model"
  - "#modality/eeg"
  - "#method/self_supervised"
created: "2026-08-04"
updated: "2026-08-07"
status: "3C paper-aligned LOSO running (foundation_3c_loso_paper_v1); DSGNet compare after results"
---

# 端到端基础模型路线计划 v1 —— 5 个 S4/DINO-DualCD 模型 × 跨被试 × 双数据集分开训练

> **一句话状态（2026-08-07）**：WBCIC **三分类 11 人**已按 DSGNet 论文（SHUv5）协议对齐
> （LOSO；train ses1–2 / val ses3 / test=留一被试全 session），5 个 foundation 正在
> `foundation_3c_loso_paper_v1` 重跑。对标论文 DSGNet Acc **0.6856**。DSGNet 本仓复现仍
> deferred。二分类协议仍待确认。

---

## 1. 任务变更（学长 2026-08-04 指示）

| 维度 | 变更前（Phase 3） | 变更后（本路线） |
|:---|:---|:---|
| 主线 | 跨 session 修复：Oracle 裁决 → T3A → 在线适应 | **先简化：不做在线学习，直接做端到端模型** |
| 模型 | EEGNet / DeepConvNet / FBCNet（+ 待接入预训练模型） | **学长指定的 5 个模型**（S4ERP + 4 个 DINO-DualCD 变体） |
| 泛化轴 | 跨 session（同被试不同天） | **跨被试（subject-independent）** |
| 数据集 | WBCIC-SHU + SHU 2022 | 同样两个，**分开训练，绝不合并**（58ch vs 32ch） |
| checkpoint | 每 cell 一个 | **只保留 best + last 两个**，且**支持断点续跑** |

**Phase 0–2c 的已完成结果与 Phase 3 的 TTA 后端一律保留不动**（`code/tta/`、
`4_experiments/*/tta/`、`PHASE3_ROUTE_PLAN.md`）。Phase 3 状态从「当前主线」改为
**paused（未废弃）**：本路线跑出端到端 backbone 后，Phase 3 的 Oracle/T3A 正好可以接在它上面。

---

## 2. 融合成果（2026-08-04 已完成，可运行）

### 2.1 文件映射：学长的包 → 本项目

| 学长包内文件 | 本项目位置 | 说明 |
|:---|:---|:---|
| `s4_layers.py` / `pooling.py` / `encoders.py` / `losses.py` / `models.py` | `code/models/eeg_foundation/`（同名） | 模型数学**未改**；4 处小改动逐条记录在 `code/models/eeg_foundation/README.md` §4 |
| （无） | `code/models/eeg_foundation/adapter.py` | **新增**：`{logits, features, confidence}` 契约 + `[B,C,T]` 输入 + DualCD 训练钩子 + per-trial 归一化 |
| （无） | `code/models/registry.py`（扩展） | 5 个模型可按名构建，和 EEGNet 家族同一个入口 |
| （无） | `code/configs/models/{s4erp,dualcd_s4_pos,dualcd_s4_timepatch,dualcd_s4_flatten,dualcd_transformer}.yaml` | 每个变体一份结构超参 |
| `train_template.py` | **未移植** → `code/training/e2e_trainer.py` | 模板的 `.npy` 加载 / 70-15-15 随机划分不符合本项目 manifest + 跨被试协议；其训练逻辑（AdamW + cosine + clip 4.0 + DualCD 三步）已在新训练器中保留 |
| （无） | `code/experiments/cross_subject_protocols.py` | **新增**：被试级划分（LOSO / subject k-fold / holdout）+ 泄漏断言 + 双 checkpoint 评测 |
| （无） | `code/runners.py::run_foundation_cross_subject` | 注册为 `PHASE_RUNNERS["foundation_cross_subject"]` |
| （无） | `code/configs/experiments/{foundation_cross_subject,shu_foundation_cross_subject}.yaml` | WBCIC / SHU 各一份，**分开跑** |
| `README.md` / `USAGE_GUIDE.md` / `comparison.md` | 内容并入 `code/models/eeg_foundation/README.md` + 各 model YAML 注释 | 参数量表、选型建议、超参建议都保留 |

### 2.2 5 个模型（参数量为**实测值**，`n_times=1000`, `n_classes=2`）

| registry 名 | 骨架 | 池化 | DualCD | 参数量 @58ch (WBCIC) | 参数量 @32ch (SHU) | feature_dim | 默认 lr |
|:---|:---|:---|:---:|---:|---:|---:|---:|
| `s4erp` | S4 | flatten | ✗ | 1,370,882 | 944,898 | 62336 | 1e-3 |
| `dualcd_s4_pos` | S4 | attention | ✓ | 3,168,772 | 2,316,804 | 128 | 1e-4 |
| `dualcd_s4_timepatch` | S4 | temporal bin | ✓ | 4,480,386 | 3,628,418 | 1536 | 1e-4 |
| `dualcd_s4_flatten` | S4 | flatten | ✓ | 66,861,186 | 66,009,218 | 62336 | 1e-4 |
| `dualcd_transformer` | Transformer | flatten | ✓ | 67,900,034 | 67,048,066 | 62336 | 1e-4 |

`patch_num = (1000-28)//2+1 = 487`；flatten 变体的参数几乎全在 DINO projection head
（`62336×512`），S4 骨架本身只有几十万参数。

**与学长表格的对照（诚实版）**：学长表给的是 906K / 1.99M / 3.30M / 65.8M / 66.8M。
两个 flatten 变体和 Transformer 在 32ch/1000/2 类下**吻合到 0.4% 以内**（说明那张表就是按我们这个
trial 长度算的）；但 `dualcd_s4_pos`（2.32M vs 1.99M）和 `dualcd_s4_timepatch`（3.63M vs 3.30M）
高出 10–16%——这两行看起来是在别的配置下量的（学长包 `comparison.md` 里 C=21/T=170 的 ERP 配置
恰好是 1.85M / 3.27M）。**移植没有问题**，上表才是我们两个数据集的实测真值；
`dualcd_s4_timepatch` 还会随 config 里的 bin 数变化。

### 2.3 针对运动想象做的 config 级适配（不是改模型）

1. **DINO 两个频带视图改成 mu/beta（8–13 / 13–30 Hz）**，替换原 ERP 默认的 4–12 / 12–30 Hz。
   （原包 README 让用户设 `model.multi_view.low_freq`，但代码里没人读那个属性 —— 已修成真参数。）
2. **`dualcd_s4_timepatch` 的时间分箱必须显式给**：原默认是 ERP 的 0–750 ms，而我们的 trial 是
   4 s，会把 487 个 patch 里 ~90% 塞进最后一个 bin。两个 config 都设
   `[0, 500, 1000, 1500, 2000, 3000, 4000]` ms（6 bins，前段密，对应 ERD 起始）。
3. **归一化 = per-trial z-score**（原包推荐），在数据加载时做一次；per-trial 是 fit-free 的，
   **不存在训练/测试统计量泄漏**。

### 2.4 已验证（CPU，31 个测试全过）

| 测试文件 | 覆盖内容 | 结果 |
|:---|:---|:---|
| `tests/foundation/test_eeg_foundation_contract.py` | 5 模型 × {58ch, 32ch} forward 契约、3D/4D 等价、loss 有限且可反传、teacher EMA 与 prototype 真的更新、teacher 全程 frozen、归一化 fit-free、1000 点下 feature_dim 与参数量对齐学长表格 | **19 passed** |
| `tests/foundation/test_cross_subject_protocol.py` | LOSO/k-fold/holdout 划分互斥且覆盖全被试、split_seed 决定性、每 cell 只写 best+last、完成的 cell 重跑被跳过、被中断的 cell 从 epoch N+1 续跑、**换了划分/超参后拒绝续跑**、**测试被试的 trial 从未进入 train/val loader**、通道数不匹配立刻报错 | **13 passed** |

另外 `python code/run.py --dry-run` 两个 config 均 `runnable_now: true`。

### 真实数据验证（SHU，CPU，2026-08-04）

用真实 manifest 跑通了「加载 → 被试池化 → 划分 → 训练 → 存双 checkpoint」：

* 6 被试各 469–492 trial（5 session 池化）。**注意不是 500**：SHU 每 session 90–100 trial 不等
  （作者剔了坏 trial），所以**类别不保证 50/50，必须报 balanced accuracy**。
* fold 0 = train 3 / val 1 / test 2 被试，互斥；`s4erp` 实测 **944,898 参数**、`feature_dim=62336`。
* `best.pt` 3.8 MB（纯权重）、`last.pt` 11.4 MB（权重 + optimizer + RNG + history），可正常读回。
* **单 epoch 734 s（CPU，47 步 ≈ 15 s/步）**，瓶颈是 ShallowNet 的空间卷积（128 通道 × 1000 点）。
  ⟹ **CPU 只能做 2 epoch 级别的最小 smoke，正式 smoke 和全量必须上 GPU。**
* GPU smoke 已按规矩走 Slurm 提交：`sbatch -J e2e_smoke -t 00:45:00 --mem=32G
  scripts/slurm/shu_gpu.sbatch code/configs/experiments/shu_foundation_cross_subject.yaml
  --models s4erp,dualcd_s4_pos --folds 4 --folds-subset 0 --max-subjects 8 --max-epochs 2 ...`
  （job 35295，排队中）。它会给出真实的单 epoch 耗时与显存占用，用来定 epoch 预算和 batch size。

### 内存注意事项（全量前必看）

被试数据加载一次后常驻内存：WBCIC 148 session × 200 trial × 58 × 1000 × 4B ≈ **6.9 GB**
（SHU ≈ 1.6 GB）。训练集**不做拷贝**——用 `ConcatDataset` 拼 per-subject `TensorDataset`，
而不是 `torch.cat`（后者每个 cell 会再复制约 5.5 GB）。所以 WBCIC 全量的常驻内存约 7 GB + 模型/激活，
Slurm 申请 **`--mem=32G`** 比较稳妥。

---

## 3. 怎么跑

```bash
# 0) 看计划（不训练）
python code/run.py --dry-run --config code/configs/experiments/foundation_cross_subject.yaml

# 1) 极小 CPU smoke（6 被试 / 1 fold / 2 epoch），输出隔离到 *_smoke 目录
python code/run.py --config code/configs/experiments/shu_foundation_cross_subject.yaml \
    --models s4erp --split-protocol kfold_subject --folds 3 --folds-subset 0 \
    --max-subjects 6 --max-epochs 2 --batch-size 32 --num-workers 0 --device cpu \
    --out outputs/experiments/shu/foundation_cross_subject_smoke \
    --ckpt-dir checkpoints/shu/foundation_cross_subject_smoke

# 2) GPU 节点先 smoke 再全量（Slurm + mi_torch_cu118；禁登录节点跑重活）
python code/run.py --config code/configs/experiments/foundation_cross_subject.yaml --device cuda

# 3) 断点续跑：**重复执行完全相同的命令即可**。已完成的 cell 跳过，
#    被中断的 cell 从 last.pt 的下一个 epoch 继续。要重训加 --no-resume。
```

常用开关：`--models`（跑哪几个模型）、`--split-protocol {loso,kfold_subject,holdout}`、
`--folds N`、`--folds-subset 0,1`、`--seeds`、`--monitor`、`--max-epochs`、`--no-resume`。

---

## 4. checkpoint 与断点续跑契约（学长两条硬要求的落地）

每个 **cell =（模型 × fold × seed）** 只产出两个权重文件，别的一个都不写：

| 文件 | 内容 | 用途 |
|:---|:---|:---|
| `<ckpt>/best.pt` | 验证集 `monitor`（默认 macro_f1）最好那一 epoch 的权重 + 该 epoch 的验证指标 | 主报告数字 |
| `<ckpt>/last.pt` | 最后一个 epoch 的权重 **+ optimizer + scheduler + RNG + history + best 记录** | 报告 final-epoch 数字 **且**用于续跑 |

* 两个 checkpoint 都会在每个留出被试上评测：CSV 里 `accuracy...ece` 来自 `best.pt`，
  `last_accuracy...last_auc` 来自 `last.pt`。
* **原子写入**（先写 `.tmp` 再 `os.replace`）：作业被 kill 在写盘瞬间也不会留下半个坏文件。
* **完成标记**是 `cells/<cell>/result.json`；有它才算这个 cell 做完（符合 AGENTS.md §7.8
  「文件落盘才算 done」）。
* **每 epoch 重设种子** `epoch_seed_base + epoch`：第 k 个 epoch 的数据顺序与「有没有被中断过」
  无关，所以续跑结果和一口气跑完可比。
* **配置漂移守卫**：每个 cell 存一个 `cell_signature`（划分被试名单 + data_dims + model_params +
  训练配方的 sha256）。如果改了 `--folds` / `--max-subjects` / 模型超参却复用同一个输出目录，
  续跑会**直接报错**而不是悄悄拿旧划分训出来的权重继续跑。要换设置就换 `--out`，或者加 `--no-resume`。

---

## 5. 跨被试协议：现状是「默认值可跑，但等学长拍板」

### 5.1 config 里现在的默认值（**不是**已确认协议）

| 项 | 默认 | 依据 |
|:---|:---|:---|
| 划分 | `kfold_subject`, `n_folds=5`, `split_seed=0` | MOABB `CrossSubjectEvaluation(n_splits=5)` 的一等公民选项；比 LOSO 省 ~7.6× 算力，且每个被试仍恰好被测一次 |
| 验证集 | `val_mode=subjects`，从训练被试里留 15% 作验证被试 | SHU 数据集作者自己的 CSA 就是留 3 个源被试做验证；**模型选择完全不看测试被试** |
| session | 一个被试的所有 ok session 池化 | EDAPT 在 WBCIC-SHU 上就是把被试的 session 拼起来 |
| 归一化 | per-trial z-score | 原包推荐；fit-free 无泄漏 |
| epoch | 100，patience 25，monitor=macro_f1 | 原包建议 s4erp 100–300 / DualCD 200–300，先取下界 |
| seeds | `[0]` | 协议定了再扩到 5 seeds（对齐 Phase 1 标准） |

### 5.2 文献调研结论（完整版见 `inbox/cross_subject_protocol_research.md`）

* **两个数据集都没有官方跨被试划分**，两篇数据集论文自己也都没跑零样本跨被试。
* **WBCIC-SHU 2025 的跨被试文献实际上只有一篇**：EDAPT（J Neural Eng 2026），用
  **2-fold 被试划分（50% 训练被试 / 50% 测试被试）**，零样本跨被试 accuracy
  **EEGNet 0.81 / DeepConvNet 0.85 / ShallowConvNet 0.82 / ATCNet 0.71**。这是我们唯一的对标锚点。
* **SHU 2022 有地板效应**：作者自己的跨 session 只有 53.7%，与 chance（51.4–53.7%）无显著差异
  （p>0.05）；且**没有任何可验证的已发表 SHU 零样本 LOSO 数字**。跨被试落在 52–55% 属于预期，
  不是 pipeline bug —— 这与本项目 Phase 1/2b 的 SHU 结果一致。
* WBCIC-SHU 有个重要混淆：session 1→3 的 within-session accuracy 从 81.77% 升到 88.90%，
  是**被试学会做 MI**（技能习得），不是纯漂移。池化三个 session = 把三种技能水平混进训练分布。

### 5.3 待学长拍板的问题（详见 `4_experiments/CROSS_SUBJECT_PROTOCOL_MEMO.md`）

1. LOSO（对标文献、贵 7.6×）还是 5-fold subject-grouped（推荐默认）？
2. 严格零样本，还是允许在目标被试上做少量校准（EDAPT: 20 trial warm-up + 50 trial 滑窗）？
3. session 池化，还是只用 session 1 / 做 session 分层？
4. 要不要加 Euclidean Alignment 作为一条 arm（文献明确推荐为跨被试标准步骤）？
5. epoch 预算：100 还是 200–300？（直接决定总算力）
6. 5 个模型是否全跑，还是先跑 `s4erp` + `dualcd_s4_pos` 看趋势？（后两个 66M 参数最贵）
7. SHU 近 chance 时怎么处理：如实报负结果 / 只留 adaptation 条件 / SHU 退出跨被试分析？

---

## 6. 硬约束

1. **协议未确认前，不得把任何一次 full run 的数字当成结果**写进论文/汇报（可以写「预跑」）。
2. **两个数据集绝不合并**：58ch vs 32ch，各自基准。`load_subject_data` 会对通道数不符直接报错。
3. **测试被试的标签只用于最后算指标**；验证集只来自训练被试。代码里有断言 + 专门的泄漏测试。
4. **GPU 只走 Slurm + `mi_torch_cu118`**；登录节点只允许极小 smoke。
5. **不覆盖任何 `*_v1` 已完成产物**（Phase 0–2c、Phase 3 A0）。本路线的新产物走
   `outputs/experiments/{wbci_shu,shu}/foundation_cross_subject_v1/` 与
   `checkpoints/{wbci_shu,shu}/foundation_cross_subject_v1/`，smoke 走 `*_smoke`。
6. **不改学长的模型数学**；任何改动必须记进 `code/models/eeg_foundation/README.md` §4。
7. **`dualcd_s4_flatten` / `dualcd_transformer` 显存不够时先降 `train.batch_size`，不要动 `d_model`**
   （动了就和学长表格的参数量对不上）。

---

## 7. 下一步

| 步骤 | 状态 | 说明 |
|:---|:---|:---|
| 融合 5 个模型 + 训练器 + 协议 + config + 测试 | **done (2026-08-04)** | 本文件 §2 |
| 文献调研（跨被试怎么设） | **done** | `inbox/cross_subject_protocol_research.md` |
| 写给学长的协议讨论备忘 | **done** | `4_experiments/CROSS_SUBJECT_PROTOCOL_MEMO.md` |
| **发给学长确认协议** | **待用户执行** | 7 个问题，见 §5.3；备忘 `4_experiments/CROSS_SUBJECT_PROTOCOL_MEMO.md` |
| GPU smoke（SHU，1 fold × {s4erp, dualcd_s4_pos}） | **submitted, queued** | Slurm job 35295；出结果后把单 epoch 耗时/显存填进本节 |
| 全量跑 + summarize + AI 分析报告 | pending | 结果区 `4_experiments/{wbci_shu,shu}/foundation_cross_subject/` |
| （之后）把 Phase 3 的 Oracle/T3A 接到端到端 backbone 上 | future | Phase 3 只是暂停，不是废弃 |
