#!/bin/bash
# V4a sbatch — V3 + reward weight tweak (task_reward_w 0.5→0.7, disc 0.3→0.1).
# Goal: weaken AMP demo distribution pull → policy follows v_cmd more directly.
# Task class: HumanoidImResAMPVCmdV3 (unchanged). Only learning yaml differs.
# Roadmap: 01_research_docs/260502_personalized_diverse_motion_roadmap.md

#SBATCH -J phc_res_amp_vcmd_v4a
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

SNAP_DIR="exp_config/forward_walking/$(date +%y%m%d)_RES_AMP_VCMD_V4a_server"
if [[ ! -d "$SNAP_DIR" ]]; then
    mkdir -p "$SNAP_DIR"
    cp phc/data/cfg/env/env_im_res_amp_vcmd_v3.yaml          "$SNAP_DIR/" 2>/dev/null || true
    cp phc/data/cfg/learning/im_res_amp_vcmd_v4a_server.yaml "$SNAP_DIR/" 2>/dev/null || true
    cp phc/env/tasks/humanoid_im_res_amp_vcmd_v3.py          "$SNAP_DIR/" 2>/dev/null || true
    cp phc/learning/res_amp_network.py                       "$SNAP_DIR/" 2>/dev/null || true
    cp phc/learning/res_amp_agent.py                         "$SNAP_DIR/" 2>/dev/null || true
    cp train_phc_res_amp_vcmd_v4a.sh                         "$SNAP_DIR/" 2>/dev/null || true
    echo "[snapshot] saved to $SNAP_DIR"
fi

python phc/run.py \
    --task HumanoidImResAMPVCmdV3 \
    --cfg_env phc/data/cfg/env/env_im_res_amp_vcmd_v3.yaml \
    --cfg_train phc/data/cfg/learning/im_res_amp_vcmd_v4a_server.yaml \
    --motion_file sample_data/amass_walking_3clips_seamless_60s_v9.pkl \
    --headless \
    --num_envs 512 \
    --episode_length 600 \
    --max_iterations 20000 \
    --minibatch_size 8192 \
    --no_log

echo "Job ended at $(date)"
