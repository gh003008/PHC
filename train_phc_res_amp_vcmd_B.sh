#!/bin/bash
# Stage 2 sbatch SLOT B — parallel run on idx2 with seed=1.
# Spec: same as Slot A (train_phc_res_amp_vcmd.sh). Slot B exists for
# seed redundancy: PPO + AMP residual learning has noisy convergence,
# and two seeds let us pick the better checkpoint at eval time.
#
# Originally targeted idx1 but job 4056 OOM-killed during motion-data
# loading peak (15500MB MaxMemPerNode is partition cap). Switched to
# idx2 — same A5000, same 15G cap, fresh cgroup. Slot A on idx0 with
# identical mem cap runs fine; idx1 attempt caught a transient peak.
#
# Submit:  sbatch train_phc_res_amp_vcmd_B.sh
# Monitor: squeue -u $USER  / tail -F logs/phc_res_amp_vcmd_B_<jobid>.out

#SBATCH -J phc_res_amp_vcmd_B
#SBATCH -p idx2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:idx2:1
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
#
# Resume: --epoch 500 loads output/ResAMPVCmd_B_00000500.pth and continues
# from that point (Slot B was cancelled at Ep ~600 to free up an idx slot;
# we restart from the clean Ep 500 checkpoint to avoid relying on the
# non-aligned "best" snapshot at ResAMPVCmd_B.pth).
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
    --epoch 500 \
    --no_log

echo "Job B ended at $(date)"
