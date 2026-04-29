#!/bin/bash
# Run this ONCE on server1 to:
#   1. Submit S18 and S19 sbatch jobs (will start at 2026-05-01 06:00 via --begin)
#   2. Schedule scancel of S16/S17 at 06:00 in case they're still running
#
# Usage: bash exp_config/forward_walking/260501_VIC4_VCMD_S18_S19/queue_at_6am.sh

set -e
cd ~/PHC

echo "=== Submitting S18 (idx0, --begin=2026-04-30T06:00:00) ==="
sbatch exp_config/forward_walking/260501_VIC4_VCMD_S18_S19/train_S18_gpu0.sh

echo "=== Submitting S19 (idx2, --begin=2026-04-30T06:00:00) ==="
sbatch exp_config/forward_walking/260501_VIC4_VCMD_S18_S19/train_S19_gpu2.sh

echo "=== Scheduling kill of S16/S17 at 06:00 today via 'at' ==="
echo 'scancel -n vic4_vcmd_S16; scancel -n vic4_vcmd_S17' | at 06:00

echo ""
echo "=== Queue state ==="
squeue -u $USER

echo ""
echo "=== 'at' jobs ==="
atq
