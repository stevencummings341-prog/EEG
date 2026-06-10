#!/bin/bash
# =====================================================================
# 全量预处理质检 + 与官方 derivatives 对比（CPU 作业）。
# 读 153×(npz+mat)、算 PSD/可分性、出 CSV/JSON/MD + 图，必须在计算节点跑。
# 提交： sbatch scripts/slurm/compare_quality_cpu.sbatch
#   可选位置参数：$1=processed_manifest  $2=official_root  $3=output_dir
# =====================================================================
#SBATCH -J mi_qc_compare
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err
#SBATCH -p gpu2node
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=125090599@link.cuhk.edu.cn

set -euo pipefail

PROJECT_ROOT="/share/home/yuan/SYX/eeg-mi-online"
PROC_ROOT="/share/workspace2/moto_imagination/WBCIC_SHU/processed/eog_ecg_clean"
MANIFEST="${1:-$PROC_ROOT/processed_manifest.csv}"
OFFICIAL_ROOT="${2:-/share/workspace2/moto_imagination/WBCIC_SHU/derivatives/2C dataset_processeddata}"
OUTPUT_DIR="${3:-$PROC_ROOT/qc_vs_derivatives}"

cd "$PROJECT_ROOT"
echo "========================================================"
echo "Node    : $(hostname)  |  Start: $(date)"
echo "Manifest: $MANIFEST"
echo "Official: $OFFICIAL_ROOT"
echo "Output  : $OUTPUT_DIR"
echo "========================================================"

module purge
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mi_torch

python -u scripts/compare_processed_quality.py \
  --processed-manifest "$MANIFEST" \
  --official-root "$OFFICIAL_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --max-example-sessions 6

echo "--------------------------------------------------------"
echo "Done: $(date)"
echo "========================================================"
