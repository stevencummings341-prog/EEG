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
#SBATCH --mail-user=b23020028@njupt.edu.cn   # 来自 run_test.sh，按需修改

set -euo pipefail

PROJECT_ROOT="/share/home/yuan/SYX/eeg-mi-online"
CONFIG_PATH="${1:-configs/preprocess.yaml}"

cd "$PROJECT_ROOT"
echo "========================================================"
echo "Project : $PROJECT_ROOT"
echo "Config  : $CONFIG_PATH"
echo "Node    : $(hostname)  |  Start: $(date)"
echo "========================================================"

module purge
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mi_torch

# -u 关闭缓冲，日志实时可见。
python -u scripts/preprocess_all.py --config "$CONFIG_PATH"

echo "--------------------------------------------------------"
echo "Done: $(date)"
echo "========================================================"
