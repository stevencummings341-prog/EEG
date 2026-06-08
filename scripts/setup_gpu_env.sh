#!/bin/bash
# =====================================================================
# GPU env fix & verify: clone mi_torch -> mi_torch_cu118, install cu118 PyTorch,
# verify CUDA, and (only if CUDA is available) run a SMALL sanity train.
#
# Runs on COMPUTE NODES via srun (NOT the login node). Does NOT touch the shared
# `mi_torch` env (creates a separate clone). Does NOT start formal 41/10 training.
#
# Because the clone (~11G over NFS) + pip download are I/O/network-bound and need NO GPU,
# the work is split into two phases so a GPU is only held for the parts that need it:
#   MODE=clone_install : conda clone + cu118 pip install   (run on a CPU srun)
#   MODE=verify_sanity : CUDA verify + small sanity train  (run on a GPU srun)
#   MODE=all           : everything in one allocation       (default)
#
# Usage:
#   bash scripts/setup_gpu_env.sh <new_env> <mode>
#   e.g.  bash scripts/setup_gpu_env.sh mi_torch_cu118 clone_install
#         bash scripts/setup_gpu_env.sh mi_torch_cu118 verify_sanity
# =====================================================================
set -u

NEW_ENV="${1:-mi_torch_cu118}"
MODE="${2:-all}"
PROJECT_ROOT="/share/home/yuan/SYX/eeg-mi-online"

echo "=================================================================="
echo "NODE: $(hostname) | MODE: ${MODE} | NEW_ENV: ${NEW_ENV} | START: $(date)"
echo "=================================================================="

echo "=== nvidia-smi (informational; absent on CPU allocations) ==="
nvidia-smi || true

# Conda available without 'module' (use the known base, like our other srun jobs).
source /share/software/anaconda3/2024.10/etc/profile.d/conda.sh

# --------------------------------------------------------------------- #
# Phase 1: clone + cu118 install
# --------------------------------------------------------------------- #
if [ "${MODE}" = "all" ] || [ "${MODE}" = "clone_install" ]; then
    if conda env list | awk '{print $1}' | grep -qx "${NEW_ENV}"; then
        echo "[clone] env '${NEW_ENV}' already exists -> skip clone"
    else
        echo "[clone] conda create -n ${NEW_ENV} --clone mi_torch -y"
        conda create -n "${NEW_ENV}" --clone mi_torch -y
        echo "[clone] exit=$?"
    fi

    conda activate "${NEW_ENV}" || { echo "[FATAL] cannot activate ${NEW_ENV}"; exit 2; }
    echo "=== active env: ${CONDA_DEFAULT_ENV} | python: $(which python) ==="

    echo "=== torch BEFORE cu118 install ==="
    python -c "import torch; print(torch.__version__, torch.version.cuda)" || true

    echo "=== pip install cu118 torch torchvision torchaudio ==="
    pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
    echo "[pip] exit=$?"
fi

# --------------------------------------------------------------------- #
# Phase 2: CUDA verify + small sanity (needs a GPU allocation)
# --------------------------------------------------------------------- #
if [ "${MODE}" = "all" ] || [ "${MODE}" = "verify_sanity" ]; then
    conda activate "${NEW_ENV}" || { echo "[FATAL] cannot activate ${NEW_ENV}"; exit 2; }
    echo "=== active env: ${CONDA_DEFAULT_ENV} | python: $(which python) ==="

    echo "=== CUDA VERIFY ==="
    python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"

    cd "${PROJECT_ROOT}"
    CUDA_OK=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    echo "[sanity] cuda_available=${CUDA_OK}"
    if [ "${CUDA_OK}" = "True" ]; then
        echo "=== SANITY TRAIN on GPU (small: 3 subjects, 3 epochs) ==="
        python -u scripts/sanity_train.py \
            --split splits/cap_eegnet_4110_seed2026.json \
            --n-subjects 3 --epochs 3 \
            --out outputs/sanity_check/sanity_check_metrics_gpu.json
        echo "[sanity] exit=$?"
    else
        echo "[sanity] CUDA not available -> SKIP GPU sanity (no formal training started)"
    fi
fi

echo "=================================================================="
echo "DONE (${MODE}): $(date)"
echo "=================================================================="
