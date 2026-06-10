# BACKUP_AND_RECOVERY.md — anti-data-loss scheme

> Why this exists: earlier in the project important code/docs lived only in the working tree
> (uncommitted) and a tooling hiccup dropped them. And experiment **results** (`outputs/`,
> `checkpoints/`) are **gitignored**, so even a perfect commit history does NOT protect them.
> This doc defines a 4-layer safety net + a routine checklist so nothing can silently disappear.

## 1. What git manages vs. what it does NOT

**Tracked by git (recoverable from any commit / bundle):**
- `src/`, `scripts/`, `configs/`, `docs/`, `AGENTS.md`, `.cursor/rules/`, `requirements.txt`, etc.

**Gitignored — NOT in git (see `.gitignore`):** must be backed up separately:
- `outputs/*` — all run artifacts (CSVs, figures, reports, splits, manifests, job-id files).
- `checkpoints/*` — model weights (large; optional to back up).
- `logs/*` — Slurm logs.
- `manifests/*.csv|json`, `splits/*.json`, and all `*.npz/*.pt/*.mat/*.npy/*.zip/...` binaries.

So: **commit protects code+docs; a separate archive must protect `outputs/` results.**

## 2. The 4 layers

1. **Commit** every code/doc change (especially the "soul files": `AGENTS.md`, `.cursor/rules/*`).
2. **Tag** each milestone (e.g. `milestone-step2-alignment-complete`) so you can jump back fast.
3. **`git bundle`** the whole repo history into one file (a portable snapshot of all commits+tags;
   survives even if `.git/` is damaged).
4. **Archive the key `outputs/`** (CSVs / reports / figures / configs / job-ids) as a `.tar.gz`,
   since git does not track them.

Plus a **`BACKUP_MANIFEST.md`** in the backup dir recording what was backed up (sizes + sha256).

## 3. Where backups live

- Authorized backup dir (this user explicitly allowed writing here): **`/share/home/yuan/SYX/backups/`**.
- Home has a **512 GiB quota** (`quota -s`). The git bundle (code/docs only) + the results tar are
  small (~tens of MB) and fine in home. **Large/long-term or checkpoint backups → `/share/workspace2`**
  (44 TiB) to avoid the home quota.
- IO-heavy compression is best run on the **`storge`** node (10.26.1.74), not login01.
- Never back up *into* the read-only dataset tree under `/share/workspace2/.../WBCIC_SHU`.

## 4. How to create a backup (copy-paste)

```bash
cd /share/home/yuan/SYX/eeg-mi-online
STAMP=$(date +%Y-%m-%d)
BK=/share/home/yuan/SYX/backups
mkdir -p "$BK"

# (precondition) clean tree + a milestone tag
git status --short                      # should be empty
git tag                                 # confirm the milestone tag exists

# Layer 3 — git bundle (ALL branches + tags) = portable full history snapshot
git bundle create "$BK/eeg-mi-online_git_${STAMP}.bundle" --all
git bundle verify "$BK/eeg-mi-online_git_${STAMP}.bundle"   # sanity check

# Layer 4 — archive the key results (gitignored). Exclude bulky checkpoints + the
# 11G of alignment checkpoints; keep CSVs/reports/figures/splits/configs/job-ids.
tar czf "$BK/eeg-mi-online_results_${STAMP}.tar.gz" \
  outputs/analysis/session_drift_v1 \
  outputs/experiments/baseline_v1 \
  outputs/experiments/alignment_baseline_v1

# record what was backed up
{ echo "# BACKUP_MANIFEST ($STAMP)"; echo;
  echo "## git"; git log -1 --oneline; git tag --points-at HEAD;
  echo; echo "## files"; ls -la "$BK"/*"${STAMP}"*;
  echo; echo "## sha256"; sha256sum "$BK"/*"${STAMP}"*; } > "$BK/BACKUP_MANIFEST.md"
```

Notes:
- The bundle is small because `outputs/checkpoints/logs` are gitignored (not in git history).
- Checkpoints (`checkpoints/alignment_baseline_v1/` ≈ 11 GB) are **not** archived by default; they
  are re-creatable by re-running. To keep them, tar separately into `/share/workspace2`.

## 5. How to restore

**Restore code/docs from the bundle** (into a fresh clone, e.g. if `.git/` is broken):
```bash
git clone /share/home/yuan/SYX/backups/eeg-mi-online_git_<STAMP>.bundle eeg-mi-online-restored
cd eeg-mi-online-restored && git log --oneline -5 && git tag    # history + milestones present
```
Or fetch into an existing repo: `git fetch <bundle> '*:*'`.

**Roll back to a milestone:** `git checkout milestone-step2-alignment-complete` (or `git switch -c
fix <tag>`).

**Restore results** (gitignored) from the tar:
```bash
cd /share/home/yuan/SYX/eeg-mi-online
tar xzf /share/home/yuan/SYX/backups/eeg-mi-online_results_<STAMP>.tar.gz   # restores outputs/...
```

## 6. Routine checklist (do this every time)

**Before starting work / a new task**
- `git status` — if not clean, stop and ask before running anything new.

**Before submitting a big Slurm run**
- Commit code + configs first (so a mid-run crash can't lose them), THEN `sbatch`.

**After finishing a meaningful step**
- Update docs (`PROGRESS.md`, `EXPERIMENT_LOG.md`, status page) → `git add` → `git commit`.
- If it's a milestone: `git tag <name>` and run the §4 backup (bundle + results tar + manifest).

**Periodically / before risky changes**
- Re-run §4 to refresh `/share/home/yuan/SYX/backups/`.

## 7. Git identity note

This environment has no `user.name/email` configured and the rules forbid editing git config.
Commit with a one-off override reusing the repo's existing author:
```bash
git -c user.name="yuan" -c user.email="b23020028@njupt.edu.cn" commit -m "..."
```
(Confirm the author with `git log -1 --format='%an %ae'`.)
