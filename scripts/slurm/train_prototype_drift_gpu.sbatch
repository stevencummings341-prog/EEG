#!/bin/bash
# Phase 2c prototype-drift GPU job: ONE model + ONE seed over all eligible
# subjects and all directed cross-session pairs. Submit 3 models x 5 seeds = 15
# jobs (see scripts/slurm/submit_prototype_drift_full.sh).
#
# Usage:
#   sbatch -J proto_drift__eegnet_s0 scripts/slurm/train_prototype_drift_gpu.sbatch eegnet 0
# Positional args: $1 = model (eegnet|deepconvnet|fbcnet), $2 = seed (int).
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err
#SBATCH -p gpu2node
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -t 16:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=125090599@link.cuhk.edu.cn

set -euo pipefail

PROJECT_ROOT="/share/home/yuan/SYX/eeg-mi-online"
MI_ENV="${MI_ENV:-mi_torch_cu118}"
MODEL="${1:?model required (eegnet|deepconvnet|fbcnet)}"
SEED="${2:?seed required (int)}"

cd "$PROJECT_ROOT"
mkdir -p logs/slurm
echo "Project: $PROJECT_ROOT"
echo "Env    : $MI_ENV"
echo "Model  : $MODEL | Seed: $SEED"
echo "Node   : $(hostname) | Start: $(date)"

source /share/software/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate "$MI_ENV"

nvidia-smi || true
python - <<'PY'
import torch
print(f"[cuda-check] torch={torch.__version__} cuda_build={torch.version.cuda} cuda_avail={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit(2)
PY

python -u code/run.py \
  --config code/configs/experiments/phase2c_prototype_drift.yaml \
  --models "$MODEL" --seeds "$SEED" --device cuda

echo "Done: $(date)"
