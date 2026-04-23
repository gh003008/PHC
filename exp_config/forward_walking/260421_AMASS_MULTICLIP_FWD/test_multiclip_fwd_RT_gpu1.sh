#!/bin/bash
#SBATCH -J testg_multiclip_fwd_RT
#SBATCH -p idx1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:idx1:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 00:15:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate phc
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

cd ~/PHC

python phc/run.py \
  --task HumanoidImVICCmdMultiClip \
  --cfg_env exp_config/forward_walking/260421_AMASS_MULTICLIP_FWD/env_im_walk_vic_multiclip_fwd_RT.yaml \
  --cfg_train exp_config/forward_walking/260421_AMASS_MULTICLIP_FWD/im_walk_vic_multiclip_fwd_RT.yaml \
  --num_envs 1 --test --epoch -1 --no_virtual_display --headless
