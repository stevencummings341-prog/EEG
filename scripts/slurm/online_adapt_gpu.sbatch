#!/bin/bash
# =====================================================================
# 在线 test-then-update 适应（GPU 作业）。从 run_test.sh 改写，适配 mi_torch。
# 提交： sbatch scripts/slurm/online_adapt_gpu.sbatch [configs/online_adaptation.yaml]
# =====================================================================
#SBATCH -J mi_online
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err
#SBATCH -p gpu2node
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=125090599@link.cuhk.edu.cn

set -euo pipefail

PROJECT_ROOT="/share/home/yuan/SYX/eeg-mi-online"
CONFIG_PATH="${1:-configs/online_adaptation.yaml}"

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

nvidia-smi || true
python - <<'PY'
import torch
ok = torch.cuda.is_available()
print(f"[cuda-check] torch={torch.__version__} cuda_build={torch.version.cuda} cuda_avail={ok}")
if not ok:
    print("[cuda-check][WARN] CUDA 不可用！见 docs/ENVIRONMENT.md。")
PY

python -u scripts/run_online_adaptation.py --config "$CONFIG_PATH"

echo "--------------------------------------------------------"
echo "Done: $(date)"
echo "========================================================"
