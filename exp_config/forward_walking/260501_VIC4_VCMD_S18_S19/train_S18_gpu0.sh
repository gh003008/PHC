#!/bin/bash
#SBATCH -J vic4_vcmd_S18
#SBATCH -p idx0
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:idx0:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH -t 36:00:00
#SBATCH --begin=2026-04-30T06:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Slot S18: S16 base + Fix A (knee_k 20→60) + curriculum_switch=100000

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
  --cfg_env exp_config/forward_walking/260501_VIC4_VCMD_S18_S19/env_im_walk_vic_S18.yaml \
  --cfg_train exp_config/forward_walking/260501_VIC4_VCMD_S18_S19/im_walk_vic.yaml \
  --headless --num_envs 256 --no_log \
  --experiment VIC4_VCMD_S18

echo "=== Job finished: $(date) ==="
