# AMASS Pipeline Verification — Server Run Instructions

**Date**: 2026-04-17
**Author session**: Local Claude (exolab-MS-7D56) → Server Claude (server1)
**Purpose**: Run 2 parallel VIC experiments on AMASS forward_single to verify the learning pipeline still reaches ~100% walking. Pre-requisite before returning to H5 data fixes.

---

## 1. Context (Why This Run)

### Problem history
Recent H5-based experiments topped out at ~50% walking success (best: no-VIC H5 = 605 reward / 192 steps ≈ 50% of target).
The expected near-100% was gunhee's earlier VIC_PHASE result: **932 reward / 299 steps** on `amass_isaac_walking_forward_single.pkl`.

### Two hypothesis for the regression
1. **H5 data conversion errors** (several upper-body joints mapped to zero — fixed partially via v3 remap, see `01_research_docs/260417_h5_upper_remap_findings.md` if present).
2. **Learning framework drift** (config changes since gunhee's run).

### Strategy: isolate hypothesis (2) first
Before re-running with fixed H5 data, we need to confirm the pipeline STILL reaches ~100% on gunhee's original AMASS data. If it does → hypothesis (1) is the problem. If it doesn't → framework regression needs to be fixed first.

---

## 2. What This Run Does

Two parallel jobs, both with:
- Motion: `sample_data/amass_isaac_walking_forward_single.pkl` (1 clip, 5.43s, KIT forward walking)
- MLP [1024, 1024, 512, 512], PPO + AMP, 20k epochs, num_envs=512
- VIC Stage 2 (learnable CCF), sigma_init=-1.0, phase_obs=True
- Reward curriculum switch at epoch 10k

### Job A — VIC 8-group (gunhee's exact replication)
- **exp_name**: `AMASS_VERIFY_VIC8`
- **Config**: `env_im_walk_vic_8grp.yaml` + `im_walk_vic_8grp.yaml`
- **Submit**: `sbatch exp_config/forward_walking/260417_AMASS_VERIFY/train_vic8_gpu2.sh`
- **Target GPU**: physical idx 2 (see §4 for pinning)
- **Target metric**: `av_reward ≥ 900`, `av_steps ≥ 290` at epoch 20k

### Job B — VIC 4-group (lower-body only + upper fixed at 1.0x)
- **exp_name**: `AMASS_VERIFY_VIC4`
- **Config**: `env_im_walk_vic_4grp.yaml` + `im_walk_vic_4grp.yaml`
- **Submit**: `sbatch exp_config/forward_walking/260417_AMASS_VERIFY/train_vic4_gpu3.sh`
- **Target GPU**: physical idx 3
- **Target metric**: aim for similar (~900/290); 4-group may converge cleaner since upper-body CCF is fixed and adds no exploration noise.

---

## 3. Server Claude — Run This Sequence

```bash
cd ~/PHC

# 1. Pull latest from git (this package)
git pull origin Jimin

# 2. Verify motion pkl exists
ls -la sample_data/amass_isaac_walking_forward_single.pkl
# Expect ~290 KB file

# 3. Verify configs are in place
ls exp_config/forward_walking/260417_AMASS_VERIFY/

# 4. Submit both jobs
mkdir -p logs
sbatch exp_config/forward_walking/260417_AMASS_VERIFY/train_vic8_gpu2.sh
sbatch exp_config/forward_walking/260417_AMASS_VERIFY/train_vic4_gpu3.sh

# 5. Confirm both queued/running
squeue -u $USER

# 6. Monitor (first 10 minutes — just to catch immediate errors)
tail -f logs/amass_verify_vic8_*.out
# In another terminal:
tail -f logs/amass_verify_vic4_*.out
```

Wait until you see `motion_lib_base: Loaded ...` and `Started new training run` type messages. Then the jobs run for 8–12 hours each.

---

## 4. GPU Index Pinning

**Default (Slurm-managed)**: Sbatch scripts use `--gres=gpu:1` — Slurm will assign whatever GPU is free.

**If you need to pin to physical idx 2 and 3**: Slurm's `--gres=gpu:1` maps whatever it allocates to `CUDA_VISIBLE_DEVICES=0` inside the job. Pinning to specific physical GPUs requires either:

**Option 1**: constraint-based (if Slurm is configured for per-GPU allocation):
```bash
#SBATCH --gres=gpu:a5000:1
#SBATCH --nodelist=server1
```
and check `nvidia-smi` inside the job to see which one you got.

**Option 2**: Skip Slurm, run directly (NOT recommended per SERVER_GUIDE rule, but if idx 2/3 are reserved):
```bash
# Terminal 1 (job A on GPU 2)
CUDA_VISIBLE_DEVICES=2 bash exp_config/forward_walking/260417_AMASS_VERIFY/train_vic8_gpu2.sh

# Terminal 2 (job B on GPU 3)
CUDA_VISIBLE_DEVICES=3 bash exp_config/forward_walking/260417_AMASS_VERIFY/train_vic4_gpu3.sh
```

Use tmux/screen so the sessions survive disconnection.

Decide based on GPU availability (`nvidia-smi`).

---

## 5. Monitoring & Evaluation

### Metrics to watch
- `av_reward` (rolling-mean episode reward)
- `av_steps` (episode length at 30 Hz — 300 = full 10s; 299 = near-100% success)
- `task_reward`, `disc_reward` (split)
- CCF statistics (printed in log every few hundred epochs)

### Checkpoints
Saved to `output/HumanoidIm/humanoid_smpl/AMASS_VERIFY_VIC{8,4}/`

### Decision tree after both jobs finish

| VIC8 result | VIC4 result | Conclusion |
|---|---|---|
| av_reward ≥ 900 & steps ≥ 290 | (similar) | ✅ **Pipeline healthy**. Return to H5 fixing with confidence. |
| av_reward 500–900 | (similar) | ⚠ Pipeline partially regressed. Bisect configs vs gunhee's 260307_VIC11 / 260310_VIC_CCF_ON2 snapshots. |
| av_reward < 500 | (similar) | ❌ Major framework regression. Check recent commits to `amp_agent.py`, `humanoid_im_vic.py`, `motion_lib_*.py`. |
| VIC8 ≫ VIC4 | — | 8-group captures motion better (upper CCF matters here). |
| VIC4 ≫ VIC8 | — | Upper CCF is noise on this task (4-group is cleaner). |

---

## 6. After Completion

Pull checkpoints back for local evaluation:
```bash
# On local machine
scp server1:~/PHC/output/HumanoidIm/humanoid_smpl/AMASS_VERIFY_VIC8_*/*.pth output/
scp server1:~/PHC/output/HumanoidIm/humanoid_smpl/AMASS_VERIFY_VIC4_*/*.pth output/
```

Visualize locally:
```bash
python phc/run.py \
  --task HumanoidImVIC \
  --cfg_env exp_config/forward_walking/260417_AMASS_VERIFY/env_im_walk_vic_8grp.yaml \
  --cfg_train exp_config/forward_walking/260417_AMASS_VERIFY/im_walk_vic_8grp.yaml \
  --num_envs 1 --test --epoch -1
```

Write up findings in `02_research_dev/260418_amass_pipeline_verify_result_analysis.md` (or whenever training completes).

---

## 7. Files in This Package

| File | Purpose |
|---|---|
| `README_FOR_SERVER.md` | This file |
| `env_im_walk_vic_8grp.yaml` | Env config — 8 groups, AMASS forward_single, phase_obs=True |
| `env_im_walk_vic_4grp.yaml` | Env config — 4 groups, AMASS forward_single, phase_obs=True |
| `im_walk_vic_8grp.yaml` | Learning config — exp_name `AMASS_VERIFY_VIC8` |
| `im_walk_vic_4grp.yaml` | Learning config — exp_name `AMASS_VERIFY_VIC4` |
| `train_vic8_gpu2.sh` | Sbatch script for VIC8 |
| `train_vic4_gpu3.sh` | Sbatch script for VIC4 |

---

## 8. Reference: Previous Results

| Experiment | Motion | VIC | rwd | steps |
|---|---|---|---|---|
| VIC_PHASE (gunhee) | AMASS forward_single | 8 grp + phase | **932** | **299** |
| VIC_CCF_ON2 | AMASS forward_single | 8 grp | 939 | 297 |
| VIC11 | AMASS forward_single | 8 grp | 947 | 300 |
| Cell C (recent) | H5 library | 8 grp | 392 | 130 |
| Cell D (recent) | H5 library | off | 605 | 192 |
| Cell E (recent) | H5 library | 4 grp | 301 | 100 |
| Cell F (recent) | H5 library | 4 grp + phase | 397 | 131 |

**Expected today**: `AMASS_VERIFY_VIC8` should match the top row (~932/299). If it does, we know the H5 degradation is data-side, not framework-side.
