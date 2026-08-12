---
title: "交接文档 — paper_baseline_3c_821_v1（8:2:1 跨被试统一对比 run）"
tags:
  - "#modality/eeg"
created: "2026-08-12"
updated: "2026-08-12"
status: "active"
---

# 交接文档 — `paper_baseline_3c_821_v1`

> 面向「同一集群、换到另一个账号继续跑」。只讲这一个 run，其他阶段（Phase 0–2c / Phase 3 TTA）
> 与本文无关，照旧看 `AGENTS.md`。
>
> **对应 Git commit**：`ad0c5d2`（已 push 到 `origin/main`）。新账号 `git pull` 到这个提交即代码一致。
> **旧账号项目根**（拷贝源）：`/share/home/Zihang/MI`
> **当前状态**：**已暂停**（Slurm job `37966–37973` 已 `scancel`），进度 **50/77 cell**。

---

## 1. 这个实验在跑什么

一句话：**在 WBCIC-SHU 三分类（11 被试）上，用同一套 8:2:1 跨被试划分 + 同一套论文 recipe，把 4 个已发表 baseline 和我们 3 个模型放在同一个 run 里跑完，得到一张可直接对比的表，并输出 train/val/test 三条曲线。**

| 维度 | 设定 |
|:---|:---|
| 数据 | WBCIC-SHU 3C，11 被试 × 3 session，`n_channels=58`、`n_times=1000`、`n_classes=3`、`sfreq=250` |
| 泛化轴 | **跨被试**（subject-independent），LOSO 11 折 |
| 划分 | **8:2:1（按被试）**：每折留 1 被试测试；其余 10 人里 2 人当验证、8 人当训练，每人 3 session 全用。trial 数正好 **7199 : 1800 : 900** |
| recipe | DSGNet 论文 §IV-A：Adam、lr `1e-4`、batch `128`、max `500` epoch；无 scheduler / 无 weight decay / 无 grad clip |
| 早停 | `patience=100`（**我们的偏离**，论文是固定 500 epoch；学长要求加早停） |
| 模型选择 | 只看**验证集** `macro_f1` 存 `best.pt` |
| 归一化 | `per_sample_zscore`（per-trial、fit-free，统计量不跨划分） |
| 三曲线 | 每 epoch 记 train / val / test 三条；**test 只做可视化，绝不参与 best.pt 或早停**（守卫在 `code/training/e2e_trainer.py`） |
| seed | 只 `[0]` |

**7 个模型**（`models:` 顺序）：

| 模型 | 来源 | 实测参数量 | 论文 Table II SHUv5 Acc |
|:---|:---|---:|---:|
| `eegnet_official` | [18] `vlawhern/arl-eegmodels`（Keras→Torch 1:1） | 3,523 | 0.6492 |
| `eegnex` | [20] `chenxiachan/EEGNeX`（Keras→Torch 1:1） | 59,275 | 0.6488 |
| `eeg_deformer` | [23] `yi-ding-cs/EEG-Deformer`（官方 PyTorch 原样） | 1,612,307 | 0.6529 |
| `atcnet` | [24] `Altaheri/EEG-ATCNet`（Keras→Torch 1:1） | 114,719 | 0.6834（论文里最强 baseline） |
| `dualcd_s4_flatten` | 我们（学长包） | 66,923,523 | — |
| `s4erp` | 我们（学长包） | 1,433,219 | — |
| `dualcd_transformer` | 我们（学长包） | 67,962,371 | — |

**baseline 硬规矩**：只认原作者官方仓库。没有完整官方代码的一律排除并写明理由 —— EEG-Inception [27]、MDGEEG [35]、EEG-DG [38]、DSGNet 本身（只有预览）都被排除；DSGNet 的 Acc 0.6856 / F1 0.6833 只作为**论文引用数字**，不是我们跑出来的。理由记在 `code/models/paper_baselines/README.md`。

**⚠ 这个 run 不是 DSGNet 论文划分。** 论文是「同被试的 session 3 当验证」，那一版在 `foundation_cross_subject_wbci_3c.yaml`（`run_id=foundation_3c_loso_paper_v1`，已跑完）。本 run 是学长 2026-08-09 要求的 8:2:1 按被试划分。两者都测留出被试的 3 个 session，**不要混着报**。

---

## 2. 当前进度（截至 2026-08-12 23:50）

**50 / 77 cell 完成**（77 = 7 模型 × 11 折）。

| 模型 | 完成 | 缺的 fold |
|:---|:---|:---|
| `eegnet_official` | 11/11 | — |
| `eegnex` | 11/11 | — |
| `eeg_deformer` | 11/11 | — |
| `atcnet` | 8/11 | 5, 8, 10（有 `last.pt`，可断点续） |
| `dualcd_s4_flatten` | 3/11 | 3,4,5,6,7,8,9,10 |
| `s4erp` | 3/11 | 3,4,5,6,7,8,9,10 |
| `dualcd_transformer` | 3/11 | 3,4,5,6,7,8,9,10 |

### 阶段性数字（**只含已完成折，不能当最终结果**）

```
eegnet_official      Acc 0.6636±0.1245  F1 0.6606  n=11
eegnex               Acc 0.6991±0.1293  F1 0.6968  n=11
eeg_deformer         Acc 0.6849±0.1414  F1 0.6825  n=11
atcnet               Acc 0.7273±0.0961  F1 0.7218  n=8   ← 折数不同，勿直接比
dualcd_s4_flatten    Acc 0.7137±0.0947  F1 0.7130  n=3
s4erp                Acc 0.7541±0.0901  F1 0.7538  n=3
dualcd_transformer   Acc 0.7156±0.0957  F1 0.7155  n=3
```

三个大模型只有 fold 0–2，而被试难度差异极大（sub-011 两条 arm 都只有 ~0.41）。**n 不同的模型之间不可比**，必须等 11/11 齐了再下任何结论。

### 为什么停了

原先 4 卡按 fold 分组（3+3+3+2、每组 7 模型）提交 `36928–36931`：只有 `36928` COMPLETED，`36929/30/31` 全部 **TIMEOUT@48h**。想加长墙时到 96h/72h/60h **全被 `QOSMaxWallDurationPerJobLimit` 拒**——这个集群 QOS 上限就是 **48h**。于是改成「每个未完成 fold 一个 job + `--models` 只跑缺的模型」，重新提交了 `37966–37973`；随后用户决定换账号，已全部 `scancel`。

单折耗时实测（用于排 job）：

| 模型 | h/fold（均值） | 最长 |
|:---|---:|---:|
| `dualcd_transformer` | 6.98 | 7.13 |
| `dualcd_s4_flatten` | 4.84 | 5.09 |
| `atcnet` | 2.06 | 2.90 |
| `s4erp` | 0.74 | 0.91 |
| `eeg_deformer` | 0.55 | 0.76 |
| `eegnex` | 0.46 | 0.82 |
| `eegnet_official` | 0.22 | 0.29 |

剩余总量 ≈ **107 GPU·小时**。按「一折一 job」，每个 job ≈ 12.6h（flatten+s4erp+transformer），远在 48h 内；若能 4 卡并发，约 26h 墙钟跑完。

---

## 3. 关键文件在哪（新账号 pull 后即有）

| 内容 | 路径 |
|:---|:---|
| **本实验全部参数** | `code/configs/experiments/paper_baseline_3c_821.yaml` |
| 模型结构默认超参 | `code/configs/models/{eegnet_official,eegnex,eeg_deformer,atcnet,dualcd_s4_flatten,s4erp,dualcd_transformer}.yaml` |
| 本 run 实际用的模型超参 | 同一实验 YAML 的 `model_params:` 段（**优先于** models/*.yaml） |
| 已发表 baseline 代码 + 出处 | `code/models/paper_baselines/`（含 README 说明每个模型的官方仓库与移植方式） |
| ATCNet | `code/models/atcnet/`（`_official_keras/` 是 vendored 官方 Keras，`atcnet_torch.py` 是 1:1 移植） |
| 我们的模型 | `code/models/eeg_foundation/` |
| 模型注册表 | `code/models/registry.py`（按名字构建） |
| 训练器（含三曲线 + 续跑） | `code/training/e2e_trainer.py` |
| 跨被试协议（划分/评测/跳过逻辑） | `code/experiments/cross_subject_protocols.py` |
| runner 接线 | `code/runners.py` → `run_foundation_cross_subject` |
| Slurm 提交/续跑脚本 | `scripts/slurm/submit_paper_baseline_821.sh` |
| 汇总脚本 | `scripts/summarize_cross_subject.py` |
| 三曲线绘图 | `scripts/plot_three_curves.py` |
| 论文对标锚点 | `4_experiments/wbci_shu/foundation_cross_subject/DSGNET_SHUv5_3C_ANCHOR.md` |

**不在 Git 里的**（必须拷或重建）：`outputs/`、`checkpoints/`、`logs/`、`*.local.yaml`、预处理数据。

---

## 4. 怎么把已有成果拷到新账号

新账号项目根记作 `$NEW`（例：`/share/home/<新号>/MI`）。旧账号根 = `/share/home/Zihang/MI`。

### 4.1 先对齐代码

```bash
cd "$NEW" && git pull        # 应到 ad0c5d2 或更新
git log --oneline -1
```

### 4.2 本机路径配置（**不进 Git**）

```bash
cd "$NEW"
ls code/configs/paths.local.yaml || cp code/configs/paths.example.yaml code/configs/paths.local.yaml
```

本 run 只用一个逻辑键，确认它在 `paths.local.yaml` 的 `manifests:` 段里：

```yaml
manifests:
  wbci_3c_processed_manifest: "outputs/processed/wbci_shu_3c_mat_clean/processed_manifest.csv"
```

这是**项目内相对路径**，所以只要把数据拷到 `$NEW/outputs/processed/wbci_shu_3c_mat_clean/` 就不用改配置。
（旧账号的 `paths.local.yaml` 也可以直接 `cp` 过去，其余键指向 `/share/workspace2/...` 共享盘，两个号只要都能读就没问题。）

### 4.3 三份要拷的东西

| # | 内容 | 大小 | 必要性 |
|:--|:---|---:|:---|
| A | 3C 预处理数据 `outputs/processed/wbci_shu_3c_mat_clean/` | 2.0 G | **必须**，否则没数据可训 |
| B | run 输出 `outputs/experiments/wbci_shu/paper_baseline_3c_821_v1/` | 14 M | **必须**，这是「哪些 cell 已完成」的唯一凭据，也存着已完成折的指标 + 三曲线 history |
| C | 权重 `checkpoints/wbci_shu/paper_baseline_3c_821_v1/` | 8.8 G | **可选**，见下 |

```bash
NEW=/share/home/<新号>/MI          # ← 改成真实路径
OLD=/share/home/Zihang/MI

mkdir -p "$NEW/outputs/processed" \
         "$NEW/outputs/experiments/wbci_shu" \
         "$NEW/checkpoints/wbci_shu"

# A. 数据（2.0G，必须）
rsync -ah --info=progress2 \
  "$OLD/outputs/processed/wbci_shu_3c_mat_clean/" \
  "$NEW/outputs/processed/wbci_shu_3c_mat_clean/"

# B. run 输出（14M，必须）
rsync -ah --info=progress2 \
  "$OLD/outputs/experiments/wbci_shu/paper_baseline_3c_821_v1/" \
  "$NEW/outputs/experiments/wbci_shu/paper_baseline_3c_821_v1/"

# C-精简（推荐，6M）：只拷可续跑的 ATCNet 断点，省 8.8G
rsync -ah --info=progress2 \
  --include='atcnet__fold5__seed0/***' \
  --include='atcnet__fold8__seed0/***' \
  --include='atcnet__fold10__seed0/***' \
  --include='*/' --exclude='*' \
  "$OLD/checkpoints/wbci_shu/paper_baseline_3c_821_v1/" \
  "$NEW/checkpoints/wbci_shu/paper_baseline_3c_821_v1/"
```

**为什么 C 可以精简**：三曲线 `history` 和所有指标都写在 `outputs/.../cells/*/result.json` 里（`scripts/summarize_cross_subject.py` 和 `scripts/plot_three_curves.py` **只读 `cells/`，不读权重**）。所以已完成 cell 的 `best.pt` / `last.pt` 对「跑完 + 出表 + 出图」毫无必要，8.8G 里绝大部分是两个 67M 模型 fold0–2 的 541 MB 级权重。

**什么时候要 C-完整**：以后要拿训练好的权重做别的事（可视化特征、接 TTA、重新评测、放论文补充材料）。那就整目录 rsync：

```bash
rsync -ah --info=progress2 \
  "$OLD/checkpoints/wbci_shu/paper_baseline_3c_821_v1/" \
  "$NEW/checkpoints/wbci_shu/paper_baseline_3c_821_v1/"
```

**可以不拷的**：`dualcd_s4_flatten` fold 3/4/6 的 `last.pt`（各 541 MB）。那是被 `scancel` 打断前只训了约 25 分钟的残局，拷 1.6 G 只省 25 分钟，不如让它从头训 —— 不拷不影响正确性。

### 4.4 拷完自检

```bash
cd "$NEW"
# 应为 50
find outputs/experiments/wbci_shu/paper_baseline_3c_821_v1/cells -name result.json | wc -l
# 应存在
ls outputs/processed/wbci_shu_3c_mat_clean/processed_manifest.csv
# 不该报错，且计划里已完成的 cell 会被识别
python code/run.py --dry-run --config code/configs/experiments/paper_baseline_3c_821.yaml
```

---

## 5. 新账号上怎么继续跑

```bash
cd "$NEW"
bash scripts/slurm/submit_paper_baseline_821.sh
```

脚本现在的形态：**fold 3–10 各一个 job**，每个 job 只跑该 fold 还缺的模型，`-t 48:00:00`。fold/模型清单写死在脚本顶部的 `FOLD_MODELS`，**若中途又完成了一部分，重提交前照 §6 重新数一遍并改这个表**（不改也行，只是会白排已完成的 cell —— 它们会被跳过，不会重训）。

续跑语义（`code/experiments/cross_subject_protocols.py`）：

- 有 `result.json` → **跳过**，数字不变
- 无 `result.json` 但有 `last.pt` → 从断点续（optimizer / scheduler / RNG / history / patience 全部复原）
- 两者都无 → 从头训

新账号的好处是 QOS 配额独立。旧号撞到过两个墙：

- `QOSMaxWallDurationPerJobLimit`：单 job **最长 48h**，改不了
- `QOSMaxCpuPerUserLimit`：`shu_gpu.sbatch` 每 job 要 8 CPU，实测同时只能跑 4 个 job，其余 PENDING

环境：conda env **`mi_torch_cu118`**，分区 `gpu2node`，`--gres=gpu:1`，`--mem=40G`。**禁止在登录节点跑训练。**

---

## 6. 随时查进度

```bash
cd "$NEW"
find outputs/experiments/wbci_shu/paper_baseline_3c_821_v1/cells -name result.json | wc -l   # 目标 77
squeue -u "$USER" -o '%.10i %.9P %.20j %.2t %.10M %.12l %R'
tail -5 logs/slurm/pb821_*-*.out
```

逐模型看缺哪些折：

```bash
python - <<'PY'
from pathlib import Path
root = Path('outputs/experiments/wbci_shu/paper_baseline_3c_821_v1/cells')
models = ['eegnet_official','eegnex','eeg_deformer','atcnet',
          'dualcd_s4_flatten','s4erp','dualcd_transformer']
for m in models:
    miss = [f for f in range(11)
            if not (root / f'{m}__fold{f}__seed0' / 'result.json').exists()]
    print(f'{m:22s} done {11-len(miss):2d}/11  missing {miss}')
PY
```

---

## 7. 77/77 之后做什么

```bash
python scripts/summarize_cross_subject.py --run outputs/experiments/wbci_shu/paper_baseline_3c_821_v1
python scripts/plot_three_curves.py      --run outputs/experiments/wbci_shu/paper_baseline_3c_821_v1
```

然后把最终 Acc / F1 对着论文 Table II 写进 `progress.md` + `0_docs/STATUS.md`，锚点数字更新到 `DSGNET_SHUv5_3C_ANCHOR.md`。

---

## 8. 坑（务必看完）

1. **别改 `paper_baseline_3c_821.yaml`。** 改了划分或超参又复用同一输出目录，`cell_signature` 守卫会直接报错（这是故意的，防止悄悄把两种设定的结果混进一张表）。真要改设定就换 `run_id`。
2. **`--no-resume` 会重训覆盖**，不要顺手加。
3. **test 曲线只能画图。** 任何形式的「看 test 挑 epoch / 挑模型」都作废这个 run 的可信度。
4. **两个 DualCD 用梯度累积**（`micro_batch_per_model`: flatten 32、transformer 16）保持等效 batch 128；24 G 卡直接 batch 128 会 OOM。**不要改成直接降 batch**，那会偷偷改掉 recipe。已知偏离：这两个模型的 BatchNorm 统计量按 micro-batch 算。
5. `n=3` 和 `n=11` 的模型不能比。等齐 11 折。
6. `outputs/.../runs/*.csv` 是带 `--tag-suffix` 的分片汇总，**不是**权威结果；权威只认 `cells/*/result.json`（汇总脚本也只读它）。
7. 别把 `eegnet_official`（本 run 的官方 EEGNet）和 `eegnet`（Phase 0–2c 用的项目内变体）搞混。
8. `*.local.yaml` **永远不要提交**。

---

## 9. 相关文档

| 文件 | 作用 |
|:---|:---|
| `AGENTS.md` | 项目唯一权威记忆 |
| `FOUNDATION_E2E_ROUTE_PLAN.md` | 端到端主线路线 |
| `4_experiments/CROSS_SUBJECT_PROTOCOL_MEMO.md` | 跨被试协议讨论备忘 |
| `4_experiments/wbci_shu/foundation_cross_subject/DSGNET_SHUv5_3C_ANCHOR.md` | 论文 Table II 对标锚点 |
| `code/models/paper_baselines/README.md` | 每个 baseline 的官方出处 / 移植方式 / 排除理由 |
| `SETUP.md` | 换机环境与 `*.local.yaml` 配置 |
| `progress.md` | 进度日记 |
