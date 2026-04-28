#!/bin/bash
#SBATCH -J vic4_vcmd_S13
#SBATCH -p idx3
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:idx3:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH -t 36:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Slot S13: v9 3-clip data + S10 reward + stage2 rebalance (task 0.5/disc 0.5).
# Tests if AMP disc dominance was driving joint-angle drift.

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate phc
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

echo "=== Job started: $(date) on $(hostname) ==="
nvidia-smi -L
echo "=================================================="

cd ~/PHC

python phc/run.py \
  --task HumanoidImVICCmdMultiClip \
  --cfg_env exp_config/forward_walking/260428_VIC4_VCMD_v9/env_im_walk_vic_S13.yaml \
  --cfg_train exp_config/forward_walking/260428_VIC4_VCMD_v9/im_walk_vic.yaml \
  --headless --num_envs 512 --no_log \
  --experiment VIC4_VCMD_S13

echo "=== Job finished: $(date) ==="
