#!/bin/bash
# Stage 2 sbatch — full residual + AMP + v_cmd training on server1.
# Spec: docs/superpowers/specs/2026-05-01-phc-residual-amp-vcmd-design.md §7 Stage 2
# Runs on a single A5000 (24 GB) so we get the full num_envs=512 budget.
# Expect 36-48h to reach max_epochs=20000.
#
# Submit:  sbatch train_phc_res_amp_vcmd.sh
# Monitor: squeue -u $USER  / tail -F logs/phc_res_amp_vcmd_<jobid>.out

#SBATCH -J phc_res_amp_vcmd
#SBATCH -p idx0
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:idx0:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH -t 48:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate phc
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "Job started on $(hostname) at $(date)"
nvidia-smi -L
export PYTHONUNBUFFERED=1

# A5000 has 24 GB; expandable_segments still helps with PPO mini-batch fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/PHC

# Stage 2 overrides — Stage 1 smoke yaml stays untouched, only deltas via CLI:
#   --num_envs 512               (vs smoke's 32)
#   --episode_length 600         (10s @ 60Hz; smoke uses 300)
#   --max_iterations 20000       (vs smoke's 500)
#   --minibatch_size 8192        (512 envs × 32 horizon = 16384 buffer; 16384/8192 = 2 mini-batches)
# AMP buffer sizes (200K) live in the _server learning yaml.
python phc/run.py \
    --task HumanoidImResAMPVCmd \
    --cfg_env phc/data/cfg/env/env_im_res_amp_vcmd.yaml \
    --cfg_train phc/data/cfg/learning/im_res_amp_vcmd_server.yaml \
    --headless \
    --num_envs 512 \
    --episode_length 600 \
    --max_iterations 20000 \
    --minibatch_size 8192 \
    --no_log

echo "Job ended at $(date)"
