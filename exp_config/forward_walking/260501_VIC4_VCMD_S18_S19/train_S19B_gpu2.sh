#!/bin/bash
#SBATCH -J vic4_vcmd_S19B
#SBATCH -p idx2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:idx2:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH -t 36:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Slot S19B: replaces failed S19. Aggressive A — knee_k 60→120, knee_w 0.5,
# foot_pos_w preserved at 0.3.

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
  --cfg_env exp_config/forward_walking/260501_VIC4_VCMD_S18_S19/env_im_walk_vic_S19B.yaml \
  --cfg_train exp_config/forward_walking/260501_VIC4_VCMD_S18_S19/im_walk_vic.yaml \
  --headless --num_envs 256 --no_log \
  --experiment VIC4_VCMD_S19B

echo "=== Job finished: $(date) ==="
