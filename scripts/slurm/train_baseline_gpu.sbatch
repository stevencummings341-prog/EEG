#!/bin/bash
# =====================================================================
# Baseline 训练（GPU 作业）。从 run_test.sh 改写，适配 mi_torch 环境。
# 提交： sbatch scripts/slurm/train_baseline_gpu.sbatch [configs/eegnet_baseline.yaml]
# =====================================================================
#SBATCH -J mi_baseline
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err
#SBATCH -p gpu2node
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=b23020028@njupt.edu.cn   # 来自 run_test.sh，按需修改

set -euo pipefail

PROJECT_ROOT="/share/home/yuan/SYX/eeg-mi-online"
CONFIG_PATH="${1:-configs/eegnet_baseline.yaml}"

cd "$PROJECT_ROOT"
echo "========================================================"
echo "Project : $PROJECT_ROOT"
echo "Config  : $CONFIG_PATH"
echo "Node    : $(hostname)  |  Start: $(date)"
echo "========================================================"

module purge
module load cuda/11.8 anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mi_torch

# --- CUDA fail-fast 检查（见 docs/ENVIRONMENT.md：当前 mi_torch 是 CPU-only torch）---
nvidia-smi || true
python - <<'PY'
import torch
ok = torch.cuda.is_available()
print(f"[cuda-check] torch={torch.__version__} cuda_build={torch.version.cuda} cuda_avail={ok}")
if not ok:
    print("[cuda-check][WARN] CUDA 不可用！mi_torch 可能是 CPU-only torch，"
          "请先按 docs/ENVIRONMENT.md 安装 cu118 版 torch，再做大规模训练。")
PY

python -u scripts/train_baseline.py --config "$CONFIG_PATH"

echo "--------------------------------------------------------"
echo "Done: $(date)"
echo "========================================================"
