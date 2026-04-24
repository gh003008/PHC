#!/bin/bash
#SBATCH -J kit425_personal_C
#SBATCH -p idx2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:idx2:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# KIT_425 PERSONAL Slot C RETIME_CMDW
# Same as B but with cmd_tracking_w=0.3 as a redundant safety net.
# If retime alone is enough (B) -> C should match. If retime has subtle issues,
# cmd_reward picks up the slack and C > B.

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
  --cfg_env exp_config/forward_walking/260424_KIT425_PERSONAL_VCMD/env_im_walk_vic_kit425_personal_C.yaml \
  --cfg_train exp_config/forward_walking/260424_KIT425_PERSONAL_VCMD/im_walk_vic_kit425_personal.yaml \
  --headless --num_envs 512 --no_log \
  --experiment KIT425_PERSONAL_C

echo "=== Job finished: $(date) ==="
