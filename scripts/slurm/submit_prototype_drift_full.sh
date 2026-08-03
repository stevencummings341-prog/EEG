#!/bin/bash
# Submit the FULL Phase 2c prototype-drift run: 3 models x 5 seeds = 15 GPU jobs,
# then one CPU summarizer with --dependency=afterany on all of them. Records job
# ids to outputs/experiments/prototype_drift_v1/full_job_ids.txt.
#
# Usage: bash scripts/slurm/submit_prototype_drift_full.sh
set -euo pipefail

PROJECT_ROOT="/share/home/yuan/SYX/eeg-mi-online"
cd "$PROJECT_ROOT"
mkdir -p logs/slurm outputs/experiments/wbci_shu/prototype_drift_v1

MODELS=(eegnet deepconvnet fbcnet)
SEEDS=(0 1 2 3 4)
GPU_SBATCH="scripts/slurm/train_prototype_drift_gpu.sbatch"
SUM_SBATCH="scripts/slurm/summarize_prototype_drift_cpu.sbatch"
IDS_FILE="outputs/experiments/wbci_shu/prototype_drift_v1/full_job_ids.txt"

: > "$IDS_FILE"
echo "# Phase 2c full submission $(date)" >> "$IDS_FILE"

GPU_IDS=()
for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    jname="proto_drift__${model}_s${seed}"
    jid=$(sbatch --parsable -J "$jname" "$GPU_SBATCH" "$model" "$seed")
    echo "GPU  $jname  ->  $jid" | tee -a "$IDS_FILE"
    GPU_IDS+=("$jid")
  done
done

DEP=$(IFS=:; echo "${GPU_IDS[*]}")
sum_jid=$(sbatch --parsable --dependency=afterany:"$DEP" "$SUM_SBATCH")
echo "SUM  proto_drift_sum  ->  $sum_jid  (afterany:${DEP})" | tee -a "$IDS_FILE"

echo ""
echo "Submitted ${#GPU_IDS[@]} GPU jobs + 1 summarizer."
echo "Job ids recorded in: $IDS_FILE"
