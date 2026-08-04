#!/bin/bash
# Generic CPU job: run code/run.py for ONE config (Phase 0 drift, or --summarize).
# No GPU. Uses mi_torch_cu118. Project root auto-detected from this script location.
#
# Usage (drift diagnostic):
#   sbatch -J shu_p0 scripts/slurm/shu_cpu.sbatch code/configs/experiments/shu_phase0_drift_diagnostic.yaml
# Usage (summarize a phase):
#   sbatch -J shu_p1_sum scripts/slurm/shu_cpu.sbatch code/configs/experiments/shu_phase1_baseline.yaml --summarize
# Optional mail: sbatch --mail-user=you@example.com -J ...
# Positional: $1 = config path. $2.. = extra args forwarded verbatim to code/run.py.
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err
#SBATCH -p gpu2node
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH --mail-type=FAIL,END

set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
CONFIG="${1:?config path required}"
shift || true

cd "$PROJECT_ROOT"
mkdir -p logs/slurm
echo "Project: $PROJECT_ROOT | Env: $MI_ENV | Node: $(hostname) | Start: $(date)"
echo "Config : $CONFIG"
echo "Extra  : $*"

mi_activate_conda

python -u code/run.py --config "$CONFIG" "$@"

echo "Done: $(date)"
