#!/bin/bash
# =====================================================================
# 全量预处理（CPU 作业）。从 run_test.sh 改写，适配本项目与 mi_torch 环境。
# 集群只有 GPU 分区，但本作业不申请 GPU（无 --gres），只用 CPU。
# 提交： sbatch scripts/slurm/preprocess_cpu.sbatch [configs/preprocess.yaml]
# =====================================================================
#SBATCH -J mi_preprocess
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err
#SBATCH -p gpu2node
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=125090599@link.cuhk.edu.cn

set -euo pipefail

PROJECT_ROOT="/share/home/yuan/SYX/eeg-mi-online"
CONFIG_PATH="${1:-configs/preprocess.yaml}"
PATHS_PATH="${2:-configs/paths.yaml}"

cd "$PROJECT_ROOT"
echo "========================================================"
echo "Project : $PROJECT_ROOT"
echo "Paths   : $PATHS_PATH"
echo "Config  : $CONFIG_PATH"
echo "Node    : $(hostname)  |  Start: $(date)"
echo "========================================================"

module purge
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mi_torch

# -u 关闭缓冲，日志实时可见。跑全量 51x3（mode=eog_ecg_clean，每 session 一个 .npz）。
# 小规模可加 --subjects/--sessions/--limit/--tag（见 preprocess_all.py）。
python -u scripts/preprocess_all.py --paths "$PATHS_PATH" --config "$CONFIG_PATH"

echo "--------------------------------------------------------"
echo "Done: $(date)"
echo "========================================================"
