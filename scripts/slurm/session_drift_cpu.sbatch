#!/bin/bash
# =====================================================================
# Cross-session drift diagnostic (direction A) — CPU job (no GPU).
# numpy/scipy/sklearn only; uses the mi_torch env (CPU torch is fine, unused).
# All extra args after the script are forwarded to run_session_drift.py.
#
# Submit:
#   # smoke test (2 subjects):
#   sbatch scripts/slurm/session_drift_cpu.sbatch --subjects 1,2
#   # full run (all ok subjects):
#   sbatch scripts/slurm/session_drift_cpu.sbatch
#
# Logs -> logs/slurm/. NEVER run the analysis on the login node.
# =====================================================================
#SBATCH -J mi_drift
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err
#SBATCH -p gpu2node
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=125090599@link.cuhk.edu.cn

set -euo pipefail

PROJECT_ROOT="/share/home/yuan/SYX/eeg-mi-online"
MI_ENV="${MI_ENV:-mi_torch}"

cd "$PROJECT_ROOT"
echo "========================================================"
echo "Project : $PROJECT_ROOT"
echo "Env     : $MI_ENV"
echo "Args    : $*"
echo "Node    : $(hostname)  |  Start: $(date)"
echo "========================================================"

module purge
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$MI_ENV"

python -u scripts/analysis/run_session_drift.py --config configs/session_drift.yaml "$@"

echo "--------------------------------------------------------"
echo "Done: $(date)"
echo "========================================================"
