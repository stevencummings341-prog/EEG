#!/bin/bash
# Generic SHU GPU job: run code/run.py for ONE config with optional passthrough args.
# Heavy/GPU training MUST go through this (never on the login node). Uses mi_torch_cu118.
#
# Usage:
#   sbatch -J shu_p1 scripts/slurm/shu_gpu.sbatch code/configs/experiments/shu_phase1_baseline.yaml --models eegnet --seeds 0
# Positional: $1 = config path. $2.. = extra args forwarded verbatim to code/run.py.
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
CONFIG="${1:?config path required (code/configs/experiments/shu_phase*.yaml)}"
shift || true

cd "$PROJECT_ROOT"
mkdir -p logs/slurm
echo "Project: $PROJECT_ROOT | Env: $MI_ENV | Node: $(hostname) | Start: $(date)"
echo "Config : $CONFIG"
echo "Extra  : $*"

source /share/software/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate "$MI_ENV"

nvidia-smi || true
python - <<'PY'
import torch
print(f"[cuda-check] torch={torch.__version__} cuda_build={torch.version.cuda} cuda_avail={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit(2)
PY

python -u code/run.py --config "$CONFIG" --device cuda "$@"

echo "Done: $(date)"
