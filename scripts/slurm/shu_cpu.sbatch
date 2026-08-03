#!/bin/bash
# Generic SHU CPU job: run code/run.py for ONE config (Phase 0 drift diagnostic, or
# any --summarize aggregation). No GPU. Submitted on gpu2node without --gres (CPU-only
# slice) because the cluster exposes only GPU partitions. Uses mi_torch_cu118.
#
# Usage (drift diagnostic):
#   sbatch -J shu_p0 scripts/slurm/shu_cpu.sbatch code/configs/experiments/shu_phase0_drift_diagnostic.yaml
# Usage (summarize a phase):
#   sbatch -J shu_p1_sum scripts/slurm/shu_cpu.sbatch code/configs/experiments/shu_phase1_baseline.yaml --summarize
# Positional: $1 = config path. $2.. = extra args forwarded verbatim to code/run.py.
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err
#SBATCH -p gpu2node
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=125090599@link.cuhk.edu.cn

set -euo pipefail

PROJECT_ROOT="/share/home/yuan/SYX/eeg-mi-online"
MI_ENV="${MI_ENV:-mi_torch_cu118}"
CONFIG="${1:?config path required}"
shift || true

cd "$PROJECT_ROOT"
mkdir -p logs/slurm
echo "Project: $PROJECT_ROOT | Env: $MI_ENV | Node: $(hostname) | Start: $(date)"
echo "Config : $CONFIG"
echo "Extra  : $*"

source /share/software/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate "$MI_ENV"

python -u code/run.py --config "$CONFIG" "$@"

echo "Done: $(date)"
