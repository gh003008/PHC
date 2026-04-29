#!/bin/bash
#SBATCH -J vic4_vcmd_S17
#SBATCH -p idx2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:idx2:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH -t 36:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Slot S17: S14 base + bounded v_cmd ramp (max_accel=0.5 m/s²)
# multiclip_resample_on_cycle=True so the ramp actually fires

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
  --cfg_env exp_config/forward_walking/260429_VIC4_VCMD_S16_S17/env_im_walk_vic_S17.yaml \
  --cfg_train exp_config/forward_walking/260429_VIC4_VCMD_S16_S17/im_walk_vic.yaml \
  --headless --num_envs 384 --no_log \
  --experiment VIC4_VCMD_S17

echo "=== Job finished: $(date) ==="
