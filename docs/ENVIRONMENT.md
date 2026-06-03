# Environment

Conda env for this project: **`mi_torch`** (`/share/home/yuan/.conda/envs/mi_torch`).

Activate:

```bash
module load anaconda3            # or: source /share/software/anaconda3/2024.10/etc/profile.d/conda.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mi_torch
```

## Verified versions (as of project setup)

| Package | Version | Notes |
| --- | --- | --- |
| python | 3.10.18 | |
| torch | 2.6.0 | **CPU-only build — see warning below** |
| mne | 1.10.0 | modern API (not the dataset code's 0.22) |
| numpy | 2.2.5 | numpy 2.x — watch for 3rd-party incompat |
| scipy | 1.15.3 | `scipy.io.loadmat` reads the paper `.mat` |
| scikit-learn | 1.7.1 | |
| pandas | 2.2.3 | |
| h5py | 3.14.0 | |
| einops | 0.8.1 | |
| pyyaml | (installed) | config loading |
| braindecode | NOT installed | optional; we reimplement baselines |

## ⚠️ Critical: torch is CPU-only

`python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"`
currently prints `None False`. `torch.version.cuda is None` means the installed
wheel is the **CPU-only** build — it will not use the GPU even on a GPU node.

Before any real GPU training, install a CUDA-enabled torch matching the cluster's
`cuda/11.8` module. Do this **inside an interactive GPU job**, not on the login node,
and confirm with the maintainer before changing a shared env. Example (verify the
exact version/index against the cluster driver first):

```bash
# inside: srun -p gpu2node --gres=gpu:1 -c 8 --mem=16G -t 01:00:00 --pty bash
conda activate mi_torch
pip install --index-url https://download.pytorch.org/whl/cu118 \
    torch torchvision torchaudio
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

If `mi_torch` must stay CPU-only (shared with others), create a dedicated env
(e.g. `mi_torch_cu118`) instead and document it here.

> This decision is left to the user; the project will not silently mutate a shared
> conda environment. The GPU sbatch scripts print a CUDA check at startup so a
> CPU-only misconfiguration fails fast and visibly.

## Optional extras (install when needed, not now)

- `braindecode` — reference baseline implementations.
- `tensorboard` / `wandb` — experiment tracking.
- `moabb`, `pyriemann` — for the Riemannian / connectivity experts (v2).

## Reproducing the env from scratch

See `requirements.txt` at the project root for the intended package set. Pin exact
CUDA-matched torch wheels there once the GPU build is chosen.
