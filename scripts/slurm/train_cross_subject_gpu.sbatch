#!/bin/bash
# =====================================================================
# 41/10 跨被试预训练（GPU 作业）。从 run_test.sh 改写，适配 mi_torch 环境。
# 提交： sbatch scripts/slurm/train_cross_subject_gpu.sbatch [configs/cross_subject.yaml]
# =====================================================================
#SBATCH -J mi_cross_subj
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err
#SBATCH -p gpu2node
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=b23020028@njupt.edu.cn   # 来自 run_test.sh，按需修改

set -euo pipefail

PROJECT_ROOT="/share/home/yuan/SYX/eeg-mi-online"
CONFIG_PATH="${1:-configs/cross_subject.yaml}"

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

python -u scripts/train_cross_subject.py --config "$CONFIG_PATH"

echo "--------------------------------------------------------"
echo "Done: $(date)"
echo "========================================================"
