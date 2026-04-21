#!/bin/bash
#SBATCH -J amass_multiclip_RT
#SBATCH -p idx1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:idx1:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH -t 20:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Multi-clip RT: 50 AMASS walking clips. v_cmd U(0.3, 0.9) picks nearest-v_nat
# clip, then per-env scale s = clamp(v_cmd / v_nat_clip, 0.9, 1.1) retimes the
# chosen clip to close the residual speed gap. Tests whether retiming on top of
# clip selection improves command tracking vs clip-selection alone (NR variant).

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
  --cfg_env exp_config/forward_walking/260421_AMASS_MULTICLIP/env_im_walk_vic_multiclip_RT.yaml \
  --cfg_train exp_config/forward_walking/260421_AMASS_MULTICLIP/im_walk_vic_multiclip_RT.yaml \
  --headless \
  --num_envs 512 \
  --no_log

echo "=== Job finished: $(date) ==="
