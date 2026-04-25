#!/bin/bash
#SBATCH -J kit425_cont_S3
#SBATCH -p idx2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:idx2:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH -t 36:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Slot S3 CYCLE_VCMD_HOP: S2 + multiclip_resample_on_cycle=True. v_cmd hops at every cycle boundary.

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
  --cfg_env exp_config/forward_walking/260426_KIT425_CONT/env_im_walk_vic_kit425_cont_S3_vcmdhop.yaml \
  --cfg_train exp_config/forward_walking/260426_KIT425_CONT/im_walk_vic_kit425_cont.yaml \
  --headless --num_envs 512 --no_log \
  --experiment KIT425_CONT_S3

echo "=== Job finished: $(date) ==="
