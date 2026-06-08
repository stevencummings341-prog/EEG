#!/bin/bash
# =====================================================================
# Within/Cross-session model comparison (GPU job).
# All extra args after the script are forwarded to train_session_models.py.
#
# Submit (examples):
#   # full within-session CV, all 4 models:
#   sbatch scripts/slurm/train_session_models_gpu.sbatch \
#       --config configs/session_model_compare.yaml \
#       --models eegnet,deepconvnet,fbcnet,cap_eegnet --protocol within
#   # full cross-session:
#   sbatch scripts/slurm/train_session_models_gpu.sbatch \
#       --config configs/session_model_compare.yaml \
#       --models eegnet,deepconvnet,fbcnet,cap_eegnet --protocol cross
#
# GPU torch env: defaults to mi_torch_cu118 (override with MI_ENV=<env>).
# Logs -> logs/slurm/. NEVER run training on the login node.
# =====================================================================
#SBATCH -J mi_sess_compare
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err
#SBATCH -p gpu2node
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=125090599@link.cuhk.edu.cn

set -euo pipefail

PROJECT_ROOT="/share/home/yuan/SYX/eeg-mi-online"
MI_ENV="${MI_ENV:-mi_torch_cu118}"

cd "$PROJECT_ROOT"
echo "========================================================"
echo "Project : $PROJECT_ROOT"
echo "Env     : $MI_ENV"
echo "Args    : $*"
echo "Node    : $(hostname)  |  Start: $(date)"
echo "========================================================"

# Use the explicit conda base (proven to work for mi_torch_cu118 in srun smoke tests).
# torch cu118 wheels ship their own CUDA runtime, so no `module load cuda` is needed.
source /share/software/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate "$MI_ENV"
echo "[env] python: $(which python) | CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-unset}"

nvidia-smi || true
python - <<'PY'
import torch
ok = torch.cuda.is_available()
print(f"[cuda-check] torch={torch.__version__} cuda_build={torch.version.cuda} cuda_avail={ok}")
if not ok:
    print("[cuda-check][WARN] CUDA unavailable! Use the cu-enabled env (see docs/ENVIRONMENT.md).")
PY

# Default args if none supplied: the THREE baseline architectures (within+cross).
# CAP-EEGNet is optional/future — pass --models cap_eegnet to include it later.
if [ "$#" -eq 0 ]; then
  set -- --config configs/session_model_compare.yaml \
         --models eegnet,deepconvnet,fbcnet --protocol both
fi

python -u scripts/train_session_models.py "$@"

echo "--------------------------------------------------------"
echo "Done: $(date)"
echo "========================================================"
