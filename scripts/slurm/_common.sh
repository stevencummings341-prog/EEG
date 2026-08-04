#!/bin/bash
# Shared helpers for Slurm jobs. Sourced by *.sbatch / submit_*.sh
# Resolves project root from this file's location (portable across machines).

_SLURM_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${_SLURM_COMMON_DIR}/../.." && pwd)"
MI_ENV="${MI_ENV:-mi_torch_cu118}"

mi_activate_conda() {
  if [ -n "${CONDA_ROOT:-}" ] && [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  elif [ -f /share/software/anaconda3/2024.10/etc/profile.d/conda.sh ]; then
    # shellcheck disable=SC1091
    source /share/software/anaconda3/2024.10/etc/profile.d/conda.sh
  elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  elif command -v conda >/dev/null 2>&1; then
    # conda already on PATH (e.g. module loaded)
    :
  else
    echo "ERROR: cannot find conda. Set CONDA_ROOT or put conda on PATH." >&2
    return 1
  fi
  conda activate "${MI_ENV}"
}

mi_cuda_failfast() {
  python - <<'PY'
import torch
print(f"[cuda-check] torch={torch.__version__} cuda_build={torch.version.cuda} cuda_avail={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit(2)
PY
}
