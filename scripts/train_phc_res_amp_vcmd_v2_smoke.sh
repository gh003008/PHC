#!/usr/bin/env bash
# V2 Stage 1 smoke training — local 4060 Ti (7.6 GB).
# Spec: docs/superpowers/specs/2026-05-02-phc-residual-amp-vcmd-v2-design.md
# Failure analysis: 02_research_dev/260502_res_amp_vcmd_failure_analysis.md
#
# All 6 V2 gates verified before this commit (see commits 56d5378, fb3f791,
# dc8565b, 7a491c5, 39e399e). Inference baseline:
#   PHC_ZERO_RESIDUAL=1 + V2 task → eps_len 1799 (full motion), reward ~1700
#   PHC_ZERO_RESIDUAL=0 + zero-init residual → eps_len 1797, reward ~1000
#
# This smoke trains for ~500 epochs (~3-6h) to verify:
#   1. Training loop boots and runs (no shape mismatches, no crashes)
#   2. eps_len stays high (>1500) — Goodharting indicator (v1's r_survive
#      caused eps_len 525 with humanoid actually falling in 24 step)
#   3. Reward grows monotonically (no Goodhart-style metric inflation)
#   4. Periodic --test inference shows residual is improving v_cmd tracking,
#      not destroying walking gait

set -euo pipefail

cd "$(dirname "$0")/.."

# Snapshot config to exp_config/forward_walking/<date>_RES_AMP_VCMD_V2/
SNAP_DIR="exp_config/forward_walking/$(date +%y%m%d)_RES_AMP_VCMD_V2"
if [[ ! -d "$SNAP_DIR" ]]; then
    mkdir -p "$SNAP_DIR"
    cp phc/data/cfg/env/env_im_res_amp_vcmd_v2.yaml         "$SNAP_DIR/"
    cp phc/data/cfg/learning/im_res_amp_vcmd_v2.yaml        "$SNAP_DIR/"
    cp phc/env/tasks/humanoid_im_res_amp_vcmd_v2.py         "$SNAP_DIR/"
    cp phc/learning/res_amp_network.py                      "$SNAP_DIR/"
    echo "[snapshot] saved to $SNAP_DIR"
fi

# Smoke params (4060 Ti 7.6 GB):
#   num_envs=32 × horizon=32 = 1024 samples/epoch
#   minibatch=512 → 2 minibatches/epoch
#   episode_length=300 (5s @ 60Hz) for varied trajectories
#   max_iterations=500 → ~3-6h training depending on physics throughput
NUM_ENVS=32
MAX_ITER=500
EP_LEN=300

# Activate conda env first (per CLAUDE.md project rule)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate phc

LOG="logs/train_v2_smoke_$(date +%y%m%d_%H%M).out"
mkdir -p logs

echo "[v2 smoke] launching at $(date)"
echo "[v2 smoke] log: $LOG"
echo "[v2 smoke] config snapshot: $SNAP_DIR"

python phc/run.py \
    --task HumanoidImResAMPVCmdV2 \
    --cfg_env phc/data/cfg/env/env_im_res_amp_vcmd_v2.yaml \
    --cfg_train phc/data/cfg/learning/im_res_amp_vcmd_v2.yaml \
    --motion_file sample_data/amass_walking_3clips_seamless_60s_v9.pkl \
    --num_envs "$NUM_ENVS" \
    --episode_length "$EP_LEN" \
    --max_iterations "$MAX_ITER" \
    --headless \
    --no_virtual_display \
    --no_log \
    2>&1 | tee "$LOG"

echo "[v2 smoke] training done at $(date)"
echo "[v2 smoke] ckpt: output/HumanoidIm/ResAMPVCmdV2/"
echo "[v2 smoke] inspect with:"
echo "    PHC_DIAG_DONE_PRINT=1 python phc/run.py \\"
echo "      --task HumanoidImResAMPVCmdV2 \\"
echo "      --cfg_env phc/data/cfg/env/env_im_res_amp_vcmd_v2.yaml \\"
echo "      --cfg_train phc/data/cfg/learning/im_res_amp_vcmd_v2.yaml \\"
echo "      --num_envs 4 --test --epoch -1 --no_virtual_display \\"
echo "      --headless --small_terrain --episode_length 100"
