#!/bin/bash
# V3 Stage 2 sbatch — v_cmd-master clip selection (vs V2's clip-master).
# Roadmap: 01_research_docs/260502_personalized_diverse_motion_roadmap.md
#
# V3 fix vs V2: at reset, sample v_cmd uniformly first, then pick the clip
# whose v_natural is closest, and force motion_lib to that clip BEFORE
# state init. This makes state ↔ reference ↔ v_cmd consistent while
# letting each (v_cmd, clip) pair be trained directly.
#
# Single A5000 on idx1 (V2 occupies idx0). num_envs=512, max 20000 epochs.
#
# Submit:  sbatch train_phc_res_amp_vcmd_v3.sh
# Monitor: squeue -u $USER  / tail -F logs/phc_res_amp_vcmd_v3_<jobid>.out

#SBATCH -J phc_res_amp_vcmd_v3
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

echo "Job started on $(hostname) at $(date)"
nvidia-smi -L
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/PHC

SNAP_DIR="exp_config/forward_walking/$(date +%y%m%d)_RES_AMP_VCMD_V3_server"
if [[ ! -d "$SNAP_DIR" ]]; then
    mkdir -p "$SNAP_DIR"
    cp phc/data/cfg/env/env_im_res_amp_vcmd_v3.yaml          "$SNAP_DIR/" 2>/dev/null || true
    cp phc/data/cfg/learning/im_res_amp_vcmd_v3_server.yaml  "$SNAP_DIR/" 2>/dev/null || true
    cp phc/env/tasks/humanoid_im_res_amp_vcmd_v3.py          "$SNAP_DIR/" 2>/dev/null || true
    cp phc/learning/res_amp_network.py                       "$SNAP_DIR/" 2>/dev/null || true
    cp phc/learning/res_amp_agent.py                         "$SNAP_DIR/" 2>/dev/null || true
    cp train_phc_res_amp_vcmd_v3.sh                          "$SNAP_DIR/" 2>/dev/null || true
    echo "[snapshot] saved to $SNAP_DIR"
fi

python phc/run.py \
    --task HumanoidImResAMPVCmdV3 \
    --cfg_env phc/data/cfg/env/env_im_res_amp_vcmd_v3.yaml \
    --cfg_train phc/data/cfg/learning/im_res_amp_vcmd_v3_server.yaml \
    --motion_file sample_data/amass_walking_3clips_seamless_60s_v9.pkl \
    --headless \
    --num_envs 512 \
    --episode_length 600 \
    --max_iterations 20000 \
    --minibatch_size 8192 \
    --no_log

echo "Job ended at $(date)"
