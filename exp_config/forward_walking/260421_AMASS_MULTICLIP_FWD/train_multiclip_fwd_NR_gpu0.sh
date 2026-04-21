#!/bin/bash
#SBATCH -J amass_multiclip_fwd_NR
#SBATCH -p idx0
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:idx0:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH -t 20:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Multi-clip FWD NR: retrieval restricted to 22 direction-verified forward-walking
# clips (compute_walking_direction_metadata.py output, v_x < 0 in PHC world-frame,
# low yaw_rate, low lateral). |v_x| ∈ [0.21, 0.99] m/s. v_cmd U(0.25, 0.85).
# No retime — isolates "clip selection only" on a clean forward subset.
# Supersedes contaminated run 3853 (cancelled) which mixed forward/backward/turning.

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
  --cfg_env exp_config/forward_walking/260421_AMASS_MULTICLIP_FWD/env_im_walk_vic_multiclip_fwd_NR.yaml \
  --cfg_train exp_config/forward_walking/260421_AMASS_MULTICLIP_FWD/im_walk_vic_multiclip_fwd_NR.yaml \
  --headless \
  --num_envs 512 \
  --no_log

echo "=== Job finished: $(date) ==="
