#!/bin/bash
# Generic GPU job: run code/run.py for ONE config with optional passthrough args.
# Heavy/GPU training MUST go through Slurm (never on the login node). Uses mi_torch_cu118.
#
# Usage:
#   sbatch -J shu_p1 scripts/slurm/shu_gpu.sbatch code/configs/experiments/shu_phase1_baseline.yaml --models eegnet --seeds 0
# Optional mail: sbatch --mail-user=you@example.com -J ...
# Positional: $1 = config path. $2.. = extra args forwarded verbatim to code/run.py.
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
CONFIG="${1:?config path required (code/configs/experiments/*.yaml)}"
shift || true

cd "$PROJECT_ROOT"
mkdir -p logs/slurm
echo "Project: $PROJECT_ROOT | Env: $MI_ENV | Node: $(hostname) | Start: $(date)"
echo "Config : $CONFIG"
echo "Extra  : $*"

mi_activate_conda
nvidia-smi || true
mi_cuda_failfast

python -u code/run.py --config "$CONFIG" --device cuda "$@"

echo "Done: $(date)"
