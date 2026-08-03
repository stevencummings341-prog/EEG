# 跨服务器拉取与运行（完整步骤）

仓库（私有）：https://github.com/stevencummings341-prog/EEG

本仓库**不含**：原始 EEG、预处理 `.npz`、模型权重、以及 `outputs/` / `logs/` / `checkpoints/` / `backup/`。  
这些需在目标机挂共享盘、从原服务器拷贝，或重新预处理生成。

---

## 0. 新机器访问私有仓库（二选一）

### 方式 A：SSH（推荐）

在**新机器**上生成钥匙（若还没有）：

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_stevencummings -C "stevencummings341-prog@$(hostname)" -N ""
cat ~/.ssh/id_ed25519_stevencummings.pub
```

1. 复制打印出的**整行**公钥  
2. 登录 GitHub 账号 **`stevencummings341-prog`** → https://github.com/settings/keys → **New SSH key** → 粘贴保存  

在新机器写入 SSH 别名（避免和别的 GitHub 账号钥匙冲突）：

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
cat >> ~/.ssh/config <<'EOF'

Host github.com-stevencummings
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_stevencummings
  IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config
```

测试：

```bash
ssh -T git@github.com-stevencummings
# 应看到: Hi stevencummings341-prog! You've successfully authenticated...
```

### 方式 B：HTTPS + Token

1. https://github.com/settings/tokens → Generate new token (classic) → 勾选 **`repo`**  
2. 克隆时用（把 `<TOKEN>` 换成真实 token，用完建议改回不含 token 的 URL）：

```bash
git clone https://stevencummings341-prog:<TOKEN>@github.com/stevencummings341-prog/EEG.git
cd EEG
git remote set-url origin https://github.com/stevencummings341-prog/EEG.git
```

---

## 1. 克隆

SSH（配合上面的 Host 别名）：

```bash
git clone git@github.com-stevencummings:stevencummings341-prog/EEG.git
cd EEG
```

若公钥已直接加到账号、且本机只有这一把 GitHub 钥匙，也可用：

```bash
git clone git@github.com:stevencummings341-prog/EEG.git
cd EEG
```

HTTPS：

```bash
git clone https://github.com/stevencummings341-prog/EEG.git
cd EEG
```

---

## 2. Python / conda 环境

```bash
conda create -n mi_torch_cu118 python=3.10 -y
conda activate mi_torch_cu118
pip install -r code/requirements.txt
```

GPU 版 PyTorch（按集群 CUDA 选择；示例 **CUDA 11.8**）：

```bash
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
```

CUDA 12.1 示例：

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

仅 CPU 时可跳过上面两行（`requirements.txt` 不钉死 torch，需自行安装匹配版本）。

---

## 3. 改机器相关路径（必做）

仓库里的绝对路径默认指向**原集群**（如 `/share/workspace2/moto_imagination/...`）。换机必须改。

| 文件 | 作用 |
|:---|:---|
| `code/configs/paths.yaml` | WBCIC-SHU raw / processed / manifest |
| `code/configs/datasets/shu.yaml` | SHU 2022 `data_dir` / `processed_root` / `manifest` |
| `code/configs/datasets/wbci_shu.yaml` | WBCIC-SHU 数据集声明（若含绝对路径一并改） |

从模板复制再改（推荐新机器第一次配置）：

```bash
cp code/configs/paths.example.yaml code/configs/paths.yaml
cp code/configs/datasets/shu.example.yaml code/configs/datasets/shu.yaml
# 用编辑器把 /CHANGE/ME/... 改成新机器上的真实路径
# 同时检查并修改 code/configs/datasets/wbci_shu.yaml
```

也可用环境变量覆盖 WBCIC raw 根：

```bash
export SHU_2C_ROOT=/your/path/WBCIC_SHU
```

**不要**把填好本机路径的 `paths.yaml` / `shu.yaml` 误推到 GitHub（会污染别人的路径）；本机改完留在本地即可。

---

## 4. 数据与权重

- **Raw（只读）**：放在 `paths.yaml` / `shu.yaml` 指向的外部目录。  
- **Processed**：需要 `processed_manifest.csv` + 各 session 的 `.npz`，或在本机重跑：
  - SHU：`python scripts/preprocess_shu.py`
  - WBCIC：见 `0_docs/`、`code/README.md` 中的预处理说明  
- **权重 / 旧实验结果**：不在 Git 里。需要时从原服务器拷贝 `checkpoints/`（或共享存储对应目录）。

目录占位（空目录结构）：

```bash
mkdir -p outputs logs checkpoints
# 仓库里已有对应 .gitkeep；大文件不会被 git 跟踪
```

---

## 5. 冒烟与开跑

```bash
conda activate mi_torch_cu118
cd /path/to/EEG   # 换成你的克隆路径

python code/run.py --dry-run --config code/configs/experiments/phase1_baseline.yaml
```

正式跑示例（需 GPU + 数据就绪）：

```bash
python code/run.py --config code/configs/experiments/shu_phase1_baseline.yaml --device cuda
python code/run.py --config code/configs/experiments/phase1_baseline.yaml --device cuda
```

更多入口见根目录 `README.md`。

---

## 6. 之后与 GitHub 同步

```bash
cd /path/to/EEG
git pull
```

若用了 SSH 别名且 remote 还不是别名，可设：

```bash
git remote set-url origin git@github.com-stevencummings:stevencummings341-prog/EEG.git
```

本地改过的 `paths.yaml` / dataset yaml 若与上游冲突，保留本机路径后再合并其它文件。

---

## 7. 原集群（本仓库推送机）备忘

- 专用钥匙：`~/.ssh/id_ed25519_stevencummings`  
- SSH Host：`github.com-stevencummings`  
- remote：`git@github.com-stevencummings:stevencummings341-prog/EEG.git`  
- 旧钥匙 `id_ed25519` 仍绑定其它 GitHub 账号（`LiaoEuan`），推本仓库请用上面的别名，不要混用。
