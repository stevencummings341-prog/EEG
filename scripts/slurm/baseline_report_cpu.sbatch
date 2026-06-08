#!/bin/bash
# =====================================================================
# Dependent overnight report job (CPU). Runs AFTER all baseline training jobs
# (submit with --dependency=afterany:<all training job ids> so it runs even if
# some training jobs fail, and reports incomplete/failed runs).
#
# Submit (example):
#   sbatch --dependency=afterany:21161:21162:...:21170 \
#          scripts/slurm/baseline_report_cpu.sbatch
#
# It checks sacct states, runs the summarizer, writes RUN_STATUS.md + the full
# report, and updates docs/EXPERIMENT_LOG.md + docs/PROGRESS.md. Logs -> logs/slurm/.
# =====================================================================
#SBATCH -J mi_base_report
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err
#SBATCH -p gpu2node
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=125090599@link.cuhk.edu.cn

set -uo pipefail   # NOT -e: we want the report to run even if a sub-step warns

PROJECT_ROOT="/share/home/yuan/SYX/eeg-mi-online"
MI_ENV="${MI_ENV:-mi_torch}"
RESULTS_DIR="${RESULTS_DIR:-outputs/experiments/session_model_compare_v1}"

cd "$PROJECT_ROOT"
echo "Project : $PROJECT_ROOT | Env: $MI_ENV | Results: $RESULTS_DIR | Node: $(hostname) | $(date)"

source /share/software/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate "$MI_ENV"
echo "[env] python: $(which python)"

python -u scripts/baseline_report.py \
    --results-dir "$RESULTS_DIR" \
    --job-ids-file "$RESULTS_DIR/overnight_job_ids.txt"

echo "Done: $(date)"
