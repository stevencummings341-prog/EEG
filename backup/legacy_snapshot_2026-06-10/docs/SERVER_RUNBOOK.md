# Server Runbook (Slurm)

操作手册：如何在学校的 Slurm 集群上安全地跑这个项目。
**最重要的一条：不要在登录节点跑训练 / 全量预处理。**

## 0. Cluster facts (verified — official "4090D 集群用户文档", http://10.26.1.75:58080/)

The cluster = **1 login node + 5 GPU nodes + 1 storage node** (all Rocky Linux 8.10), sharing one
NFS filesystem under `/share`. Access only from the campus network (VPN if off-campus).

- **login node `login01`** (10.26.1.75): the SSH entry point. **Submit + monitor jobs ONLY.** No
  compute, no GPU, no big IO here. Cheap (<~30 s) checks are fine.
- **GPU nodes `gpu01-05`** (10.26.1.101-105): **8× NVIDIA 4090D each**. No direct SSH — reach them
  ONLY via Slurm (`srun`/`sbatch`). Partitions:
  - `gpu2node` (default) = gpu01, gpu02.
  - `gpu3node` = gpu03, gpu04, gpu05.
  - TIMELIMIT effectively unlimited -> ALWAYS pass a real `-t`.
- **storage node `storge`** (10.26.1.74): SSH-able; use it for **IO-heavy** work (large file
  copy/move, **compress/decompress**, up/download via scp/rsync). NOT in Slurm; no compute there.
- **Scheduler: Slurm.** Full command set on login01 (`sinfo squeue srun sbatch scancel scontrol
  sacct`); GPU nodes are read-only (`sinfo`/`squeue`); `storge` has no Slurm.
- **Monitoring helpers (run on login01):** `slmwatch` (cluster + your quota/jobs dashboard),
  `gpuwatch` (per-GPU mem/util/temp, sampled every 60 s), `user-tools` (lists available helpers).
- **Environment Modules:** `module avail|load|unload|list|switch` (e.g. `module load cuda/11.8`
  or `cuda/12.6`). Available on login01 + all GPU nodes (NOT on storge).
- Conda base for activation: `/share/software/anaconda3/2024.10/etc/profile.d/conda.sh`.
- **Project GPU env: `mi_torch_cu118`** (torch 2.7.1+cu118, CUDA works) — use for all GPU jobs.
  The old `mi_torch` is CPU-only; do not use it for GPU.
- Shared storage (NFS, all nodes): `/share` (20 TiB SSD: home + conda), `/share/workspace` (37 TiB
  HDD), `/share/workspace2` (44 TiB HDD — our dataset lives here), `/share/workspace3` (328 TiB HDD).
- **Home-dir quota: 512 GiB quota / 2048 GiB hard limit / 14-day grace** (`quota -s` to check).
  Keep home lean; put large datasets/backups under `/share/workspace2` (or `workspace`/`workspace3`),
  not home.
- Project root: `/share/home/yuan/SYX/eeg-mi-online`.
- Dataset (READ-ONLY): `/share/workspace2/moto_imagination/WBCIC_SHU`.
- Per-user CPU/QOS cap: many submitted jobs sit `PD (QOSMaxCpuPerUserLimit)` and start as capacity
  frees — this is NORMAL.

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
source /share/software/anaconda3/2024.10/etc/profile.d/conda.sh && conda activate mi_torch_cu118
cd /share/home/yuan/SYX/eeg-mi-online
python -c "import torch; print('cuda', torch.cuda.is_available())"   # must be True on a GPU node
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
slmwatch                             # cluster + my quota/jobs dashboard (login01)
gpuwatch                             # per-GPU mem/util/temp (login01)
squeue -u $USER                      # my queued/running jobs
squeue -u $USER -o "%.18i %.12P %.20j %.8T %.10M %.6D %R"
sacct -X -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS,ReqTRES%40
scancel <jobid>                      # cancel a job (scancel -u $USER cancels all mine)
tail -f logs/slurm/<jobname>-<jobid>.out
```

## 5. GPU sanity check inside a job

The first thing every GPU job prints should be:

```bash
nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda)"
```

With env `mi_torch_cu118` this prints `cuda True` (torch 2.7.1+cu118). If `cuda_avail=False` on a
GPU node, STOP and **fail-fast** (exit non-zero) — you are likely on the old CPU-only `mi_torch`.
Never train at scale on CPU.

## 6. Etiquette on a shared cluster

- Request only the GPUs/CPUs/memory you need (`--gres=gpu:1`, `-c 8`, `--mem`).
- Always set a realistic `-t`; do not leave idle interactive sessions holding GPUs.
- Prefer many short jobs / arrays over one giant monolithic job when possible.
- Never write into the dataset dir or anywhere outside `/share/home/yuan/SYX`. Heavy IO
  (large copies, compression of backups) is best done on the `storge` node, not login01.

## 7. Backups (anti-data-loss)

git only versions code/docs (`outputs/`, `checkpoints/`, `logs/` are gitignored), so results
need a separate backup. See **`docs/BACKUP_AND_RECOVERY.md`** for the full scheme + checklist:
tag milestones, `git bundle` the repo history, and tar the key `outputs/` results to the
user-authorized `/share/home/yuan/SYX/backups/` (large/long-term backups → `/share/workspace2`).

## 8. Reference

The original submission script that this project's sbatch files are adapted from is
`/share/home/yuan/SYX/run_test.sh` (a different project, `survey` env). Our versions
target the `mi_torch_cu118` env and this project's paths.
