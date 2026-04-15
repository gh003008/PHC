#!/bin/bash
# Extract per-epoch (epoch, rwd, eps_len) from Slurm log(s) into a CSV for plotting.
# Usage:
#   ./dump_metrics_csv.sh                         # auto-detect latest log
#   ./dump_metrics_csv.sh 3609                    # specific job id
#   ./dump_metrics_csv.sh 3598 3609               # concat multiple logs in order
# Output: ~/PHC/logs/epoch_metrics.csv

OUT=~/PHC/logs/epoch_metrics.csv
echo "epoch,rwd,eps_len,frame" > "$OUT"

if [ $# -eq 0 ]; then
  JOBIDS=($(ls -t ~/PHC/logs/phc_walk_server_*.out | head -1 | grep -oP '\d+(?=\.out$)'))
else
  JOBIDS=("$@")
fi

for JOBID in "${JOBIDS[@]}"; do
  LOG=~/PHC/logs/phc_walk_server_${JOBID}.out
  if [ ! -f "$LOG" ]; then
    echo "skip: $LOG not found" >&2
    continue
  fi
  grep '^PHC_Server_Baseline' "$LOG" \
    | sed -E 's/^PHC_Server_Baseline_v1-Ep:[[:space:]]*([0-9]+)[[:space:]]+rwd:[[:space:]]*([0-9.\-]+).*frame:[[:space:]]*([0-9]+)[[:space:]]+eps_len:[[:space:]]*([0-9.\-]+).*/\1,\2,\4,\3/' \
    >> "$OUT"
done

N=$(( $(wc -l < "$OUT") - 1 ))
echo "Wrote $N rows to $OUT"
head -3 "$OUT"
echo "..."
tail -3 "$OUT"
