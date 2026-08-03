#!/bin/bash
# Phase 2c prototype-drift summarizer (CPU; submit with --dependency=afterany:<gpu_ids>).
# Merges all per-run CSVs, builds correlation tables/figures/report, and ALWAYS
# writes run_status.csv + RUN_STATUS.md listing any missing/failed cells (never faked).
#SBATCH -J proto_drift_sum
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err
#SBATCH -p gpu2node
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=125090599@link.cuhk.edu.cn

set -euo pipefail

PROJECT_ROOT="/share/home/yuan/SYX/eeg-mi-online"
MI_ENV="${MI_ENV:-mi_torch_cu118}"

cd "$PROJECT_ROOT"
mkdir -p logs/slurm
echo "Project: $PROJECT_ROOT | Env: $MI_ENV | Node: $(hostname) | Start: $(date)"

source /share/software/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate "$MI_ENV"

python -u code/run.py --summarize \
  --config code/configs/experiments/phase2c_prototype_drift.yaml

echo "Done: $(date)"
