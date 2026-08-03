# 跨服务器拉取与运行

本仓库**不含**原始 EEG、预处理 `.npz`、权重与 `outputs/` / `logs/` / `checkpoints/` / `backup/`。这些需在目标机单独准备或从共享盘挂载。

## 1. 克隆

```bash
git clone git@github.com:<OWNER>/<REPO>.git
# 或 HTTPS:
# git clone https://github.com/<OWNER>/<REPO>.git
cd <REPO>
```

## 2. 环境

建议 conda（Python 3.10），环境名可自定，例如 `mi_torch_cu118`：

```bash
conda create -n mi_torch_cu118 python=3.10 -y
conda activate mi_torch_cu118
pip install -r code/requirements.txt
# GPU（按集群 CUDA 版本选 wheel，示例 CUDA 11.8）:
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
```

## 3. 改机器相关路径（必做）

当前机上的绝对路径写在：

| 文件 | 作用 |
|:---|:---|
| `code/configs/paths.yaml` | WBCIC-SHU raw / processed / manifest |
| `code/configs/datasets/shu.yaml` | SHU 2022 `data_dir` / `processed_root` / `manifest` |
| `code/configs/datasets/wbci_shu.yaml` | WBCIC-SHU 数据集声明（若含绝对路径一并改） |

可从模板复制后改：

```bash
cp code/configs/paths.example.yaml code/configs/paths.yaml
cp code/configs/datasets/shu.example.yaml code/configs/datasets/shu.yaml
# 再编辑上述文件，指向本机数据根目录
```

也可用环境变量覆盖 WBCIC raw 根：`export SHU_2C_ROOT=/your/path/WBCIC_SHU`。

## 4. 数据

- **Raw**：只读，放在 `paths.yaml` / `shu.yaml` 指向的外部目录。
- **Processed**：需已有 `processed_manifest.csv` + per-session `.npz`，或在本机重跑预处理：
  - WBCIC：项目内预处理流水线（见 `0_docs/` / `code/README.md`）
  - SHU：`python scripts/preprocess_shu.py`

权重不在 Git 里；需要实验权重时从原服务器 `checkpoints/` 或共享存储拷贝。

## 5. 冒烟

```bash
python code/run.py --dry-run --config code/configs/experiments/phase1_baseline.yaml
```

正式跑示例（需 GPU + 数据就绪）：

```bash
python code/run.py --config code/configs/experiments/shu_phase1_baseline.yaml --device cuda
```

## 6. 与本仓库同步

```bash
git pull
# 有本地路径改动时不要覆盖 paths / dataset yaml，或改完再 commit 到本机分支
```
