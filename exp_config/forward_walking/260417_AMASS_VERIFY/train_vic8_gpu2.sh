#!/bin/bash
#SBATCH -J amass_verify_vic8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 16:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# ------------------------------------------------------------------
# AMASS_VERIFY_VIC8 — Pipeline verification run (gunhee VIC_PHASE replication)
# Target GPU: physical idx 2  (set via CUDA_VISIBLE_DEVICES if running directly)
# Motion: sample_data/amass_isaac_walking_forward_single.pkl (1 clip, 5.43s)
# Target result: ~932 reward / ~299 eps_len at epoch 20k (match gunhee's VIC_PHASE)
# ------------------------------------------------------------------

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate phc

echo "=== Job started: $(date) on $(hostname) ==="
echo "=== GPU allocated: ==="
nvidia-smi -L
echo "=================================================="
export PYTHONUNBUFFERED=1

cd ~/PHC

python phc/run.py \
  --task HumanoidImVIC \
  --cfg_env exp_config/forward_walking/260417_AMASS_VERIFY/env_im_walk_vic_8grp.yaml \
  --cfg_train exp_config/forward_walking/260417_AMASS_VERIFY/im_walk_vic_8grp.yaml \
  --headless \
  --num_envs 512

echo "=== Job finished: $(date) ==="
