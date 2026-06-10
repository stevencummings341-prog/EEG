#!/bin/bash
# Multi-source cross-session baseline GPU job: train ses-01+02 -> test ses-03.
# Extra args are forwarded to scripts/train_session_multisource.py.
# Default: three baselines, seeds 0-4, env mi_torch_cu118.
#SBATCH -J mi_multisrc
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err
#SBATCH -p gpu2node
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -t 12:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=125090599@link.cuhk.edu.cn

set -euo pipefail

PROJECT_ROOT="/share/home/yuan/SYX/eeg-mi-online"
MI_ENV="${MI_ENV:-mi_torch_cu118}"

cd "$PROJECT_ROOT"
echo "Project: $PROJECT_ROOT"
echo "Env    : $MI_ENV"
echo "Args   : $*"
echo "Node   : $(hostname) | Start: $(date)"

source /share/software/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate "$MI_ENV"

nvidia-smi || true
python - <<'PY'
import torch
print(f"[cuda-check] torch={torch.__version__} cuda_build={torch.version.cuda} cuda_avail={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit(2)
PY

if [ "$#" -eq 0 ]; then
  set -- --config code/configs/experiments/phase2a_multisource.yaml \
         --models eegnet,deepconvnet,fbcnet --seeds 0,1,2,3,4
fi

python -u code/run.py "$@"
echo "Done: $(date)"
