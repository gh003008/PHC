#!/bin/bash
# V2 Stage 2 sbatch — full residual + AMP + v_cmd training on server1.
# Spec: docs/superpowers/specs/2026-05-02-phc-residual-amp-vcmd-v2-design.md
# Failure analysis: 02_research_dev/260502_res_amp_vcmd_failure_analysis.md
#
# All 7 V2 gates verified locally (commits 56d5378, fb3f791, dc8565b,
# 7a491c5, 39e399e, 943626a). v1 failed at 5500 epoch with metric 525 / real
# 24-step fall (Goodharting via r_survive + episode-cap mismatch); V2
# inference baseline is 1799 step (full motion) at all v_cmd ∈ [0.76, 1.23].
#
# Single A5000 (24 GB), num_envs=512. Expect 36-48h to reach max_epochs=20000.
#
# Submit:  sbatch train_phc_res_amp_vcmd_v2.sh
# Monitor: squeue -u $USER  / tail -F logs/phc_res_amp_vcmd_v2_<jobid>.out

#SBATCH -J phc_res_amp_vcmd_v2
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/PHC

# Snapshot exp_config (per project rule .agent/rules/code-command.md)
SNAP_DIR="exp_config/forward_walking/$(date +%y%m%d)_RES_AMP_VCMD_V2_server"
if [[ ! -d "$SNAP_DIR" ]]; then
    mkdir -p "$SNAP_DIR"
    cp phc/data/cfg/env/env_im_res_amp_vcmd_v2.yaml          "$SNAP_DIR/" 2>/dev/null || true
    cp phc/data/cfg/learning/im_res_amp_vcmd_v2_server.yaml  "$SNAP_DIR/" 2>/dev/null || true
    cp phc/env/tasks/humanoid_im_res_amp_vcmd_v2.py          "$SNAP_DIR/" 2>/dev/null || true
    cp phc/learning/res_amp_network.py                       "$SNAP_DIR/" 2>/dev/null || true
    cp phc/learning/res_amp_agent.py                         "$SNAP_DIR/" 2>/dev/null || true
    cp train_phc_res_amp_vcmd_v2.sh                          "$SNAP_DIR/" 2>/dev/null || true
    echo "[snapshot] saved to $SNAP_DIR"
fi

# Stage 2 server params:
#   num_envs=512, horizon=32 → 16384 frames/epoch
#   minibatch=8192 → 2 minibatches/epoch
#   episode_length=600 (10s @ 60Hz)
#   max_iterations=20000 → ~36-48h on A5000
python phc/run.py \
    --task HumanoidImResAMPVCmdV2 \
    --cfg_env phc/data/cfg/env/env_im_res_amp_vcmd_v2.yaml \
    --cfg_train phc/data/cfg/learning/im_res_amp_vcmd_v2_server.yaml \
    --motion_file sample_data/amass_walking_3clips_seamless_60s_v9.pkl \
    --headless \
    --num_envs 512 \
    --episode_length 600 \
    --max_iterations 20000 \
    --minibatch_size 8192 \
    --no_log

echo "Job ended at $(date)"
