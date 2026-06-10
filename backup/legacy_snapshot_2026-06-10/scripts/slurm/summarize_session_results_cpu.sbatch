#!/bin/bash
# =====================================================================
# Summarize within/cross-session model results -> tables, figures, report (CPU job).
# pandas/matplotlib only. All extra args forwarded to summarize_session_results.py.
#
# Submit:
#   sbatch scripts/slurm/summarize_session_results_cpu.sbatch
#   sbatch scripts/slurm/summarize_session_results_cpu.sbatch \
#       --results-dir outputs/experiments/session_model_compare_v1
#
# Logs -> logs/slurm/. (Light enough that it can also be run inside any srun shell.)
# =====================================================================
#SBATCH -J mi_sess_summary
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err
#SBATCH -p gpu2node
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=125090599@link.cuhk.edu.cn

set -euo pipefail

PROJECT_ROOT="/share/home/yuan/SYX/eeg-mi-online"
MI_ENV="${MI_ENV:-mi_torch}"

cd "$PROJECT_ROOT"
echo "Project : $PROJECT_ROOT | Env: $MI_ENV | Args: $* | Node: $(hostname) | $(date)"

module purge
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$MI_ENV"

python -u scripts/summarize_session_results.py "$@"

echo "Done: $(date)"
