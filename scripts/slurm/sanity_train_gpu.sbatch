#!/bin/bash
# =====================================================================
# 训练前准备：CUDA 检查 + SHUTrialDataset smoke test + minimal CAP-EEGNet
# sanity training（2-4 source 被试，1-3 epoch）。这是 sanity，不是正式训练。
# 提交： sbatch scripts/slurm/sanity_train_gpu.sbatch [splits/cap_eegnet_4110_seed2026.json]
# =====================================================================
#SBATCH -J mi_sanity
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err
#SBATCH -p gpu2node
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=125090599@link.cuhk.edu.cn

set -euo pipefail

PROJECT_ROOT="/share/home/yuan/SYX/eeg-mi-online"
SPLIT_PATH="${1:-splits/cap_eegnet_4110_seed2026.json}"

cd "$PROJECT_ROOT"
echo "========================================================"
echo "Project : $PROJECT_ROOT"
echo "Split   : $SPLIT_PATH"
echo "Node    : $(hostname)  |  Start: $(date)"
echo "========================================================"

module purge
module load cuda/11.8 anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mi_torch

nvidia-smi || true
echo "=== CUDA CHECK ==="
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_build", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count() if torch.cuda.is_available() else 0)
if not torch.cuda.is_available():
    print("[WARN] CUDA 不可用（torch.version.cuda 为 None = CPU-only wheel），见 docs/ENVIRONMENT.md。")
PY

echo "=== DATASET SMOKE TEST ==="
python -u tests/test_dataset_smoke.py --split "$SPLIT_PATH" --n-subjects 2

echo "=== SANITY TRAINING (minimal CAP-EEGNet) ==="
python -u scripts/sanity_train.py --split "$SPLIT_PATH" --n-subjects 3 --epochs 3

echo "--------------------------------------------------------"
echo "Done: $(date)"
echo "========================================================"
