#!/bin/bash
#SBATCH -J phc_h5_novic
#SBATCH -p idx3
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:idx3:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate phc
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "Job started on $(hostname) at $(date)"
nvidia-smi -L
export PYTHONUNBUFFERED=1

cd ~/PHC
python phc/run.py \
    --task HumanoidIm \
    --cfg_env phc/data/cfg/env/env_im_walk_h5_novic.yaml \
    --cfg_train phc/data/cfg/learning/im_walk_h5_novic.yaml \
    --headless \
    --num_envs 512 \
    --no_log
