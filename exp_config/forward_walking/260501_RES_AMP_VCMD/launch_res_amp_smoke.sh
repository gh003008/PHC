#!/usr/bin/env bash
# Local smoke run on RTX 4060 Ti (7.6 GB, ~6h, num_envs=32, max_epochs=500).
# Pass criteria: spec §8 Stage 1 (av_eps_len >= 240, av_reward +30%, AMP disc 50-80%, no NaN/OOM).
#
# Boot-verification log (Ep 1-40, Apr 2026): rwd ~21, eps_len 13→17, fps ~1500, no OOM.
# Memory note: num_envs=64 OOMs on 4060 Ti at backward (frozen phc_3 PNN activation memory).
# expandable_segments helps with fragmentation across PPO mini-batch passes.

set -e
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate phc

LOG_DIR="logs/res_amp_smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python phc/run.py \
  --task HumanoidImResAMPVCmd \
  --cfg_env phc/data/cfg/env/env_im_res_amp_vcmd.yaml \
  --cfg_train phc/data/cfg/learning/im_res_amp_vcmd.yaml \
  --headless \
  --num_envs 32 \
  --no_log \
  2>&1 | tee "$LOG_DIR/train.out"
