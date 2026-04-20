#!/bin/bash
#SBATCH -J amass_cmd_B_extend
#SBATCH -p idx0
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:idx0:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH -t 20:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Baseline A: CMD_B resume to 30k epochs (ceiling check for current v_cmd design).
# Loads output/AMASS_CMD_B.pth (canonical, epoch=20001) and trains 10k more.
# save_frequency bumped 100 → 2500 per storage cap rule.

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
  --cfg_env exp_config/forward_walking/260421_AMASS_CMD_B_EXTEND/env_im_walk_vic_cmd_B.yaml \
  --cfg_train exp_config/forward_walking/260421_AMASS_CMD_B_EXTEND/im_walk_vic_cmd_B.yaml \
  --headless \
  --num_envs 512 \
  --epoch -1 \
  --no_log

echo "=== Job finished: $(date) ==="
