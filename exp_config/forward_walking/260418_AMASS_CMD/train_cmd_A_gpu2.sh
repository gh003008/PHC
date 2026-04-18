#!/bin/bash
#SBATCH -J amass_cmd_A
#SBATCH -p idx2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:idx2:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH -t 20:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# AMASS_CMD_A: velocity command conditioning, moderate tracking weight (cmd_tracking_w=0.3)
# 4-group CCF, from-scratch training (obs dim changed, parent VIC4 ckpt cannot be loaded directly)

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate phc
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "=== Job started: $(date) on $(hostname) ==="
nvidia-smi -L
echo "=================================================="
export PYTHONUNBUFFERED=1

cd ~/PHC

python phc/run.py \
  --task HumanoidImVICCmd \
  --cfg_env exp_config/forward_walking/260418_AMASS_CMD/env_im_walk_vic_cmd_A.yaml \
  --cfg_train exp_config/forward_walking/260418_AMASS_CMD/im_walk_vic_cmd_A.yaml \
  --headless \
  --num_envs 512 \
  --no_log

echo "=== Job finished: $(date) ==="
