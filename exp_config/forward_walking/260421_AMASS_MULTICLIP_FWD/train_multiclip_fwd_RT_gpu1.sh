#!/bin/bash
#SBATCH -J amass_multiclip_fwd_RT
#SBATCH -p idx1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:idx1:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH -t 20:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Multi-clip FWD RT: retrieval restricted to 22 direction-verified forward-walking
# clips. Per-env scale s = clamp(v_cmd / |v_x_clip|, 0.9, 1.1) retimes the chosen
# clip to close the residual speed gap. Supersedes contaminated run 3854.

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate phc
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "=== Job started: $(date) on $(hostname) ==="
nvidia-smi -L
echo "=================================================="
export PYTHONUNBUFFERED=1

cd ~/PHC

python phc/run.py \
  --task HumanoidImVICCmdMultiClip \
  --cfg_env exp_config/forward_walking/260421_AMASS_MULTICLIP_FWD/env_im_walk_vic_multiclip_fwd_RT.yaml \
  --cfg_train exp_config/forward_walking/260421_AMASS_MULTICLIP_FWD/im_walk_vic_multiclip_fwd_RT.yaml \
  --headless \
  --num_envs 512 \
  --no_log

echo "=== Job finished: $(date) ==="
