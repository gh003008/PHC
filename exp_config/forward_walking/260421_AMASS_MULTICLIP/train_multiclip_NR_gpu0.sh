#!/bin/bash
#SBATCH -J amass_multiclip_NR
#SBATCH -p idx0
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:idx0:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH -t 20:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Multi-clip NR: 50 AMASS walking clips, v_cmd U(0.3, 0.9) picks nearest-v_nat
# clip at env reset. No retiming (per-env scale fixed at 1.0) — isolates the
# "clip selection only" contribution. One shared policy, no network branching.

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate phc
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "=== Job started: $(date) on $(hostname) ==="
nvidia-smi -L
echo "=================================================="
export PYTHONUNBUFFERED=1

cd ~/PHC

python phc/run.py \
  --task HumanoidImVICCmdMultiClip \
  --cfg_env exp_config/forward_walking/260421_AMASS_MULTICLIP/env_im_walk_vic_multiclip_NR.yaml \
  --cfg_train exp_config/forward_walking/260421_AMASS_MULTICLIP/im_walk_vic_multiclip_NR.yaml \
  --headless \
  --num_envs 512 \
  --no_log

echo "=== Job finished: $(date) ==="
