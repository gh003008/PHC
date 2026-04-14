#!/bin/bash
#SBATCH -J phc_walk_server
#SBATCH -p idx2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:idx2:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate phc
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "Job started on $(hostname) at $(date)"
nvidia-smi -L
export PYTHONUNBUFFERED=1

cd ~/PHC
python phc/run_hydra.py \
    learning=im_walk \
    env=env_im_walk \
    exp_name=PHC_Server_Baseline_v1 \
    headless=True \
    no_log=True
