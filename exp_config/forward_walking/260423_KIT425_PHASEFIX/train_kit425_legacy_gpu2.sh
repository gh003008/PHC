#!/bin/bash
#SBATCH -J kit425_legacy
#SBATCH -p idx2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:idx2:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# KIT_425 Slot 3 LEGACY: cycle_motion=True, vic_phase_obs=True.
# Quantifies cycle_motion wrap damage vs Slot 1 MAIN. 20k epochs, 512 envs.

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
  --cfg_env exp_config/forward_walking/260423_KIT425_PHASEFIX/env_im_walk_vic_kit425_legacy.yaml \
  --cfg_train exp_config/forward_walking/260423_KIT425_PHASEFIX/im_walk_vic_kit425.yaml \
  --headless --num_envs 512 --no_log \
  --experiment KIT425_LEGACY

echo "=== Job finished: $(date) ==="
