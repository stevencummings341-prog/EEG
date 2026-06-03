# Server Runbook (Slurm)

操作手册：如何在学校的 Slurm 集群上安全地跑这个项目。
**最重要的一条：不要在登录节点跑训练 / 全量预处理。**

## 0. Cluster facts (verified)

- Scheduler: Slurm. `sbatch` / `squeue` / `scancel` / `sinfo` / `sacct` available.
- Partitions:
  - `gpu2node` (default) — 2 nodes, `gpu:8`, 128 CPU, ~773 GB RAM each.
  - `gpu3node` — 3 nodes, `gpu:8`, 128 CPU, ~773 GB RAM each.
  - TIMELIMIT is effectively unlimited -> ALWAYS pass a real `-t` (e.g. `-t 1-00:00:00`).
- Modules: `cuda/11.8`, `anaconda3`.
- Conda base for activation: `$(conda info --base)` (the system anaconda3).
- Project conda env: `mi_torch` (`/share/home/yuan/.conda/envs/mi_torch`).
- Project root: `/share/home/yuan/SYX/eeg-mi-online`.
- Dataset (READ-ONLY): `/share/workspace2/moto_imagination/WBCIC_SHU`.

## 1. What may run on the login node

OK (cheap, < ~30 s): editing, `git`, `ls`/`head`, `sinfo`, `squeue`, printing
package versions, reading one BDF header, submitting jobs.

NOT OK: training, full 51x3 preprocessing, any GPU work, anything that pins CPUs
or eats memory for minutes. Submit those with `sbatch`.

## 2. Interactive compute session (for debugging / single-session smoke tests)

```bash
# GPU debug shell (1 GPU, 2 hours):
srun -p gpu2node --gres=gpu:1 -c 8 --mem=16G -t 02:00:00 --pty bash
# then inside:
module purge && module load cuda/11.8 anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate mi_torch
cd /share/home/yuan/SYX/eeg-mi-online
python scripts/check_raw_bdf.py --help
```

CPU-only interactive (e.g. preprocessing one session): drop `--gres=gpu:1`.

## 3. Batch submission

Batch scripts live in `scripts/slurm/`. Create the log dir once:

```bash
mkdir -p /share/home/yuan/SYX/eeg-mi-online/logs/slurm
```

Submit:

```bash
cd /share/home/yuan/SYX/eeg-mi-online
sbatch scripts/slurm/preprocess_cpu.sbatch                 # uses default config
sbatch scripts/slurm/train_baseline_gpu.sbatch configs/eegnet_baseline.yaml
sbatch scripts/slurm/train_cross_subject_gpu.sbatch configs/train_cross_subject.yaml
sbatch scripts/slurm/online_adapt_gpu.sbatch configs/online_adaptation.yaml
```

Each sbatch script: sets `#SBATCH` resources, writes logs to `logs/slurm/%x-%j.out`
and `.err`, loads modules, activates `mi_torch`, `cd`s to project root, and runs the
script with the config passed as `$1` (with a sensible default).

## 4. Monitoring

```bash
squeue -u $USER                      # my queued/running jobs
squeue -u $USER -o "%.18i %.12P %.20j %.8T %.10M %.6D %R"
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS,ReqTRES%40
scancel <jobid>                      # cancel a job
tail -f logs/slurm/<jobname>-<jobid>.out
```

## 5. GPU sanity check inside a job

The first thing every GPU job prints should be:

```bash
nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda)"
```

If `cuda_avail=False` on a GPU node, STOP — the env has a CPU-only torch. Fix per
`docs/ENVIRONMENT.md` before doing a long run. Do not train at scale on CPU.

## 6. Etiquette on a shared cluster

- Request only the GPUs/CPUs/memory you need (`--gres=gpu:1`, `-c 8`, `--mem`).
- Always set a realistic `-t`; do not leave idle interactive sessions holding GPUs.
- Prefer many short jobs / arrays over one giant monolithic job when possible.
- Never write into the dataset dir or anywhere outside `/share/home/yuan/SYX`.

## 7. Reference

The original submission script that this project's sbatch files are adapted from is
`/share/home/yuan/SYX/run_test.sh` (a different project, `survey` env). Our versions
target the `mi_torch` env and this project's paths.
