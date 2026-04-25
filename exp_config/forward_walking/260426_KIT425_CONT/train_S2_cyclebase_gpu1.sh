#!/bin/bash
#SBATCH -J kit425_cont_S2
#SBATCH -p idx1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:idx1:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH -t 36:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Slot S2 CYCLE_BASE: S1 + cycle_motion=True + episode_length 3000.

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
  --cfg_env exp_config/forward_walking/260426_KIT425_CONT/env_im_walk_vic_kit425_cont_S2_cyclebase.yaml \
  --cfg_train exp_config/forward_walking/260426_KIT425_CONT/im_walk_vic_kit425_cont.yaml \
  --headless --num_envs 512 --no_log \
  --experiment KIT425_CONT_S2

echo "=== Job finished: $(date) ==="
