#!/bin/bash
# Phase 2c prototype-drift summarizer (CPU; submit with --dependency=afterany:<gpu_ids>).
# Merges all per-run CSVs, builds correlation tables/figures/report, and ALWAYS
# writes run_status.csv + RUN_STATUS.md listing any missing/failed cells (never faked).
# Optional mail: sbatch --mail-user=you@example.com ...
#SBATCH -J proto_drift_sum
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err
#SBATCH -p gpu2node
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH --mail-type=FAIL,END

set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

cd "$PROJECT_ROOT"
mkdir -p logs/slurm
echo "Project: $PROJECT_ROOT | Env: $MI_ENV | Node: $(hostname) | Start: $(date)"

mi_activate_conda

python -u code/run.py --summarize \
  --config code/configs/experiments/phase2c_prototype_drift.yaml

echo "Done: $(date)"
