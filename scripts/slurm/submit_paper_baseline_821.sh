#!/bin/bash
# Submit / resume `paper_baseline_3c_821` across GPUs.
#
# Cluster QOS hard-caps wall time at 48 h (`QOSMaxWallDurationPerJobLimit`); longer
# `-t` values are rejected. The first 4-way fold split (3+3+3+2, all 7 models) timed
# out on g2–g4 because ATCNet + DualCD with three-curve eval exceeded 48 h per group.
#
# Resume strategy (2026-08-12): one unfinished fold per job, and only the models that
# still lack `result.json` (small baselines already 11/11). Finished cells are skipped;
# interrupted ATCNet cells continue from `last.pt`.
#
# Usage:
#   bash scripts/slurm/submit_paper_baseline_821.sh
set -euo pipefail

CFG="code/configs/experiments/paper_baseline_3c_821.yaml"
RUN_DIR="outputs/experiments/wbci_shu/paper_baseline_3c_821_v1"
SBATCH="scripts/slurm/shu_gpu.sbatch"
# Max allowed by QOS on this cluster (probed 2026-08-12: 60h/72h/96h all rejected).
TIME="48:00:00"
MEM="40G"

cd "$(dirname "$0")/../.." || exit 1
mkdir -p "$RUN_DIR" logs/slurm

# fold -> comma-separated models still missing result.json
# (re-check with cells/ before editing if you re-run after more finish)
declare -A FOLD_MODELS=(
  [3]="dualcd_s4_flatten,s4erp,dualcd_transformer"
  [4]="dualcd_s4_flatten,s4erp,dualcd_transformer"
  [5]="atcnet,dualcd_s4_flatten,s4erp,dualcd_transformer"
  [6]="dualcd_s4_flatten,s4erp,dualcd_transformer"
  [7]="dualcd_s4_flatten,s4erp,dualcd_transformer"
  [8]="atcnet,dualcd_s4_flatten,s4erp,dualcd_transformer"
  [9]="dualcd_s4_flatten,s4erp,dualcd_transformer"
  [10]="atcnet,dualcd_s4_flatten,s4erp,dualcd_transformer"
)

FOLDS=(3 4 5 6 7 8 9 10)
IDS=()

for fold in "${FOLDS[@]}"; do
  models="${FOLD_MODELS[$fold]}"
  tag="f${fold}"
  jid=$(sbatch --parsable -J "pb821_${tag}" -t "$TIME" --mem="$MEM" "$SBATCH" "$CFG" \
        --folds-subset "$fold" --models "$models" --tag-suffix "$tag")
  IDS+=("$jid")
  echo "job $jid  fold=$fold  models=$models  (tag $tag)"
done

{
  echo "submitted $(date -Iseconds)"
  echo "config $CFG"
  echo "run_id paper_baseline_3c_821_v1"
  echo "note QOS MaxWall=48h; resume = 1 unfinished fold / job, remaining models only"
  echo "split 8:2:1 cross-subject (LOSO, 8 train / 2 val / 1 test subject)"
  echo "recipe DSGNet paper: Adam 1e-4, batch 128, max 500 epochs, early stop patience 100"
  echo "curves per-epoch train/val/test (test = monitoring only)"
  for i in "${!FOLDS[@]}"; do
    f="${FOLDS[$i]}"
    echo "fold ${f} models ${FOLD_MODELS[$f]} -> ${IDS[$i]}"
  done
} | tee "$RUN_DIR/parallel_job_ids.txt"

squeue -u "$USER" -o '%.10i %.9P %.20j %.2t %.10M %.12l %R'
