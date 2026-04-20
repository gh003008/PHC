#!/bin/bash
#SBATCH -J amass_retime_R1
#SBATCH -p idx1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:idx1:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH -t 20:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Baseline B — Retimed Reference R1.
# Per-env s ∈ [0.9, 1.1] retimes the walking clip. v_cmd_x = s * v_nat (=0.85 m/s).
# Reward = base imitation vs retimed teacher only (cmd_tracking_w = 0).
# AMP demo velocities are scaled per demo by s' ~ U(0.9, 1.1) to keep the
# discriminator distribution consistent with retimed rollouts.
# Fresh from-scratch training, 20k epochs, save_frequency=2500.

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate phc
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "=== Job started: $(date) on $(hostname) ==="
nvidia-smi -L
echo "=================================================="
export PYTHONUNBUFFERED=1

cd ~/PHC

python phc/run.py \
  --task HumanoidImVICCmdRetime \
  --cfg_env exp_config/forward_walking/260421_AMASS_RETIME/env_im_walk_vic_retime.yaml \
  --cfg_train exp_config/forward_walking/260421_AMASS_RETIME/im_walk_vic_retime.yaml \
  --headless \
  --num_envs 512 \
  --no_log

echo "=== Job finished: $(date) ==="
