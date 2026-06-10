#!/bin/bash
# Summarize Step-2 alignment results (CPU job; submitted with --dependency=afterany).
#SBATCH -J mi_align_sum
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err
#SBATCH -p gpu2node
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:40:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=125090599@link.cuhk.edu.cn

set -euo pipefail

PROJECT_ROOT="/share/home/yuan/SYX/eeg-mi-online"
MI_ENV="${MI_ENV:-mi_torch_cu118}"

cd "$PROJECT_ROOT"
source /share/software/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate "$MI_ENV"

python -u scripts/summarize_alignment_results.py "$@"
echo "Done: $(date)"
