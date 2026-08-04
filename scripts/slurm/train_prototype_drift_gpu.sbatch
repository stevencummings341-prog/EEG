#!/bin/bash
# Phase 2c prototype-drift GPU job: ONE model + ONE seed over all eligible
# subjects and all directed cross-session pairs. Submit 3 models x 5 seeds = 15
# jobs (see scripts/slurm/submit_prototype_drift_full.sh).
#
# Usage:
#   sbatch -J proto_drift__eegnet_s0 scripts/slurm/train_prototype_drift_gpu.sbatch eegnet 0
# Optional mail: sbatch --mail-user=you@example.com -J ...
# Positional args: $1 = model (eegnet|deepconvnet|fbcnet), $2 = seed (int).
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err
#SBATCH -p gpu2node
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -t 16:00:00
#SBATCH --mail-type=FAIL,END

set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
MODEL="${1:?model required (eegnet|deepconvnet|fbcnet)}"
SEED="${2:?seed required (int)}"

cd "$PROJECT_ROOT"
mkdir -p logs/slurm
echo "Project: $PROJECT_ROOT"
echo "Env    : $MI_ENV"
echo "Model  : $MODEL | Seed: $SEED"
echo "Node   : $(hostname) | Start: $(date)"

mi_activate_conda
nvidia-smi || true
mi_cuda_failfast

python -u code/run.py \
  --config code/configs/experiments/phase2c_prototype_drift.yaml \
  --models "$MODEL" --seeds "$SEED" --device cuda

echo "Done: $(date)"
