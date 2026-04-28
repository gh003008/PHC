#!/bin/bash
#SBATCH -J vic4_vcmd_S12
#SBATCH -p idx0
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:idx0:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 36:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Slot S12: v9 3-clip data (KIT_11 added) + S10 reward setup.
# Tests if data variance alone fixes joint-angle drift.

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
  --cfg_env exp_config/forward_walking/260428_VIC4_VCMD_v9/env_im_walk_vic_S12.yaml \
  --cfg_train exp_config/forward_walking/260428_VIC4_VCMD_v9/im_walk_vic.yaml \
  --headless --num_envs 512 --no_log \
  --experiment VIC4_VCMD_S12

echo "=== Job finished: $(date) ==="
