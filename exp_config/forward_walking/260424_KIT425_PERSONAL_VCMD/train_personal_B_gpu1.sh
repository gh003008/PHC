#!/bin/bash
#SBATCH -J kit425_personal_B
#SBATCH -p idx1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:idx1:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# KIT_425 PERSONAL Slot B RETIME_PURE
# multiclip_retime_enabled=True, cmd_tracking_w=0.0, retime_scale_range=[0.7,1.4].
# The "main" candidate: retime makes reference play at exactly v_cmd, so
# imitation reward implicitly trains v_cmd tracking. cmd_reward redundant.

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
  --cfg_env exp_config/forward_walking/260424_KIT425_PERSONAL_VCMD/env_im_walk_vic_kit425_personal_B.yaml \
  --cfg_train exp_config/forward_walking/260424_KIT425_PERSONAL_VCMD/im_walk_vic_kit425_personal.yaml \
  --headless --num_envs 512 --no_log \
  --experiment KIT425_PERSONAL_B

echo "=== Job finished: $(date) ==="
