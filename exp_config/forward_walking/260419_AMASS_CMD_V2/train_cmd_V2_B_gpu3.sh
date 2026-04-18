#!/bin/bash
#SBATCH -J amass_cmd_V2_B
#SBATCH -p idx3
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:idx3:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH -t 20:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# AMASS_CMD_V2_B: cmd_tracking_w=0.3 (stable) + personalization + wider v_cmd range (body shape variation)
# Round 1 winner's config (A, w=0.3) + personalization ON + v_cmd in [0.6, 1.4]

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
  --cfg_env exp_config/forward_walking/260419_AMASS_CMD_V2/env_im_walk_vic_cmd_V2_B.yaml \
  --cfg_train exp_config/forward_walking/260419_AMASS_CMD_V2/im_walk_vic_cmd_V2_B.yaml \
  --headless \
  --num_envs 512 \
  --no_log

echo "=== Job finished: $(date) ==="
