#!/bin/bash
#SBATCH -J vic4_vcmd_S5
#SBATCH -p idx1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:idx1:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH -t 36:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Slot S5 PRIMARY: 3 clips multiclip + retime + termDist=0.4

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
  --cfg_env exp_config/forward_walking/260427_VIC4_VCMD/env_im_walk_vic_S5.yaml \
  --cfg_train exp_config/forward_walking/260427_VIC4_VCMD/im_walk_vic.yaml \
  --headless --num_envs 512 --no_log \
  --experiment VIC4_VCMD_S5

echo "=== Job finished: $(date) ==="
