#!/bin/bash
#SBATCH -J test_kit425_personal_A
#SBATCH -p idx0
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:idx0:1
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
  --cfg_env exp_config/forward_walking/260424_KIT425_PERSONAL_VCMD/env_im_walk_vic_kit425_personal_A.yaml \
  --cfg_train exp_config/forward_walking/260424_KIT425_PERSONAL_VCMD/im_walk_vic_kit425_personal.yaml \
  --num_envs 1 --test --epoch -1 --no_virtual_display --headless \
  --experiment KIT425_PERSONAL_A
