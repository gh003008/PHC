#!/bin/bash
#SBATCH -J kill_S16_S17
#SBATCH -p idx1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH -t 00:05:00
#SBATCH --begin=2026-04-30T06:00:00
#SBATCH -o logs/%x_%j.out

echo "=== scancel S16/S17 at $(date) ==="
scancel -n vic4_vcmd_S16 2>&1 || true
scancel -n vic4_vcmd_S17 2>&1 || true
sleep 5
squeue -u $USER
echo "=== done ==="
