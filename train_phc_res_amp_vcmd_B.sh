#!/bin/bash
# Stage 2 sbatch SLOT B — parallel run on idx1 with seed=1.
# Spec: same as Slot A (train_phc_res_amp_vcmd.sh). Slot B exists for
# seed redundancy: PPO + AMP residual learning has noisy convergence,
# and two seeds let us pick the better checkpoint at eval time.
#
# Submit:  sbatch train_phc_res_amp_vcmd_B.sh
# Monitor: squeue -u $USER  / tail -F logs/phc_res_amp_vcmd_B_<jobid>.out

#SBATCH -J phc_res_amp_vcmd_B
#SBATCH -p idx1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:idx1:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH -t 48:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate phc
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "Job B (seed=1) started on $(hostname) at $(date)"
nvidia-smi -L
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/PHC

# Slot B differs from Slot A only by seed (1 vs 0) and exp_name suffix
# (so checkpoints don't clobber Slot A's). All training hparams identical.
python phc/run.py \
    --task HumanoidImResAMPVCmd \
    --cfg_env phc/data/cfg/env/env_im_res_amp_vcmd.yaml \
    --cfg_train phc/data/cfg/learning/im_res_amp_vcmd_server.yaml \
    --headless \
    --num_envs 512 \
    --episode_length 600 \
    --max_iterations 20000 \
    --minibatch_size 8192 \
    --seed 1 \
    --experiment ResAMPVCmd_B \
    --no_log

echo "Job B ended at $(date)"
