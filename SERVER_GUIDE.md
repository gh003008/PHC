# SERVER_GUIDE.md — EXOLAB Server Usage

This file guides Claude Code (and the user) on running PHC training on the **EXOLAB shared GPU server**.
Read this at the start of any session on the server.

---

## 1. Server Overview

| Item | Specification |
|------|----------------|
| **Host** | `server1` (143.248.65.114) |
| **OS** | Ubuntu 22.04 LTS |
| **Scheduler** | Slurm 23.02.7 |
| **CPUs** | 64 cores |
| **Memory** | 62 GB usable |
| **GPUs** | 4 × NVIDIA RTX A5000 (24 GB each) |
| **Conda** | System-wide at `/opt/miniconda3`, user envs at `~/.conda/envs/` |
| **Web Monitor** | http://143.248.65.114:5050 |

**Resource limits per job**: `--cpus-per-task ≤ 64`, `--mem ≤ 62G`, `--gres=gpu: ≤ 4`

---

## 2. Golden Rule: **Never run heavy jobs directly in SSH terminal.**

Always submit via Slurm `sbatch`. Interactive runs hog GPU and violate fair-use.

### Slurm Quick Reference

| Command | Description |
|----------|-------------|
| `sbatch train.sh` | Submit a job |
| `squeue` | Check running/pending jobs |
| `squeue -u $USER` | Your jobs only |
| `scontrol show job <job_id>` | Detailed job info |
| `scancel <job_id>` | Cancel a job |
| `sinfo` | Node/partition status |

---

## 3. User Home & Data Layout (`jiminyoun@server1`)

```
/home/jiminyoun/
├── PHC/                    # Git repo (gh003008/PHC.git, Jimin branch)
│   ├── data/               # SMPL + AMASS (3.5 GB, rsynced from local)
│   ├── sample_data/        # motion pkl (303 MB, rsynced)
│   ├── output/             # training outputs (server-local, NOT in git)
│   ├── log/                # logs (server-local, NOT in git)
│   └── conversation_history/  # session history (shared via git)
├── isaacgym/               # Isaac Gym (656 MB, rsynced)
├── phc_env_clean.yml       # conda env export
└── logs/                   # Slurm job logs
```

**Rule**: keep all work under `~/`. Do not touch system files.

---

## 4. Environment Setup (one-time)

```bash
# 1) Create conda env from YAML
conda env create -n phc -f ~/phc_env_clean.yml

# 2) Install PyTorch with CUDA 12.1
conda activate phc
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# 3) Install Isaac Gym (editable)
cd ~/isaacgym/python
pip install -e .

# 4) Install smpl-sim from GitHub (NOT in YAML — must install from git)
pip install git+https://github.com/ZhengyiLuo/SMPLSim.git@master

# 5) Install PHC (editable) — also installs remaining deps from requirement.txt
cd ~/PHC
pip install -e .
# If smpl-sim wasn't picked up above, also run:
# pip install -r requirement.txt

# 6) Verify GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())"
# Expect: CUDA: True GPUs: 4

# 7) Verify smpl_sim importable
python -c "import smpl_sim; print('smpl_sim OK:', smpl_sim.__file__)"
```

---

## 5. Running PHC Training on Server

### Key difference from local
Local (7.6 GB GPU) used a **shrunk** config (`env_im_walk_mpl.yaml` / `im_walk_mpl.yaml`):
- MLP [512, 256], num_envs 128, AMP buffer 10k, episode_length 150

Server (24 GB A5000) can run the **original** PHC config (`env_im_walk.yaml` / `im_walk.yaml`):
- MLP [1024, 1024, 512, 512], num_envs 512, AMP buffer 200k, episode_length 300

**Hypothesis**: config shrinkage was the root cause of walking-learning failure. Confirm by running original config on server.

### Slurm sbatch template (`~/PHC/train_phc.sh`)

```bash
#!/bin/bash
#SBATCH -J phc_walk_server
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate phc

echo "Job started on $(hostname) at $(date)"
nvidia-smi -L
export PYTHONUNBUFFERED=1

cd ~/PHC
python phc/run_hydra.py \
    learning=im_walk \
    env=env_im_walk \
    exp_name=PHC_Server_Baseline_v1 \
    num_envs=512 \
    headless=True
```

Submit:
```bash
cd ~/PHC
mkdir -p logs
sbatch train_phc.sh
squeue -u $USER
```

### Monitoring
- **Slurm log**: `tail -f logs/phc_walk_server_<jobid>.out`
- **TensorBoard**: `output/HumanoidIm/humanoid_smpl/PHC_Server_Baseline_v1_*/summaries/`
- **Web GUI**: http://143.248.65.114:5050

---

## 6. Git Workflow (cross-environment)

| Item | Strategy | Where |
|---|---|---|
| PHC code | git pull/push | `gh003008/PHC.git` Jimin branch |
| `data/`, `sample_data/`, `isaacgym/` | rsync only | `.gitignore`d |
| `output/`, `log/` | per-machine | `.gitignore`d |
| `conversation_history/` | git (shared) | in repo |

**On server**, pull latest before starting work:
```bash
cd ~/PHC && git pull origin Jimin
```

**Commit session summaries** (see `CLAUDE.md` section on conversation history):
```bash
cd ~/PHC
git add conversation_history/YYMMDD_HHMM_session.md
git commit -m "conversation_history: add <session topic>"
git push origin Jimin
```

---

## 7. Migration Status (as of 2026-04-14)

### Completed
- EXOLAB account `jiminyoun` created
- VSCode Remote-SSH `server1-jimin` configured
- Claude Code installed on server (`~/.npm-global/bin/claude`)
- SSH key generated on server, registered to GitHub account `yjm7845123` (collaborator of `gh003008/PHC`)
- `~/PHC` cloned from GitHub (Jimin branch, commits `6d94892`, `01c274d`)
- Data rsynced: `data/` (3.5 GB), `sample_data/` (303 MB), `isaacgym/` (656 MB)
- `~/phc_env_clean.yml` uploaded

### Pending
- [ ] Build conda env on server (`conda env create` + pytorch + isaacgym + phc editable)
- [ ] Verify GPU access
- [ ] Write and submit first test Slurm job (original `im_walk` config, num_envs=512)
- [ ] Evaluate whether original config resolves the walking-learning failure

---

## 8. Troubleshooting

- **`Requested node configuration is not available`**: you requested more than 64 CPU / 62 G mem / 4 GPU. Lower sbatch values.
- **Job stays PENDING**: another user's job holds the GPU index you requested. Use generic `--gres=gpu:1` instead of `--gres=gpu:idx0:1`.
- **OOM on A5000**: unlikely with original config, but if it happens, reduce `num_envs` to 256.
- **Isaac Gym import error**: make sure `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate phc` ran in the same shell, and `cd ~/isaacgym/python && pip install -e .` was done.
