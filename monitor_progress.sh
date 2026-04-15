#!/bin/bash
# Poll Slurm log for PHC training progress and report every 1000 epochs.
# Log format (for any exp):  <exp>-Ep: <N>\trwd: <R>\t...\teps_len: <L>
# Usage:
#   ./monitor_progress.sh <jobid>                   # auto-detect exp_prefix from first Ep line
#   ./monitor_progress.sh <jobid> <exp_prefix>      # pin exp_prefix manually

JOBID="${1:-}"
PREFIX="${2:-}"
if [ -z "$JOBID" ]; then
  JOBID=$(ls -t ~/PHC/logs/*.out 2>/dev/null | head -1 | grep -oP '\d+(?=\.out$)')
fi

LOG=$(ls ~/PHC/logs/*_${JOBID}.out 2>/dev/null | head -1)
if [ -z "$LOG" ]; then
  echo "monitor: log file for job ${JOBID} not found" >&2
  exit 1
fi

# Auto-detect prefix if not given
if [ -z "$PREFIX" ]; then
  for _ in $(seq 1 30); do
    PREFIX=$(grep -oP '^[A-Za-z0-9_]+(?=-Ep:)' "$LOG" 2>/dev/null | head -1)
    if [ -n "$PREFIX" ]; then break; fi
    sleep 10
  done
  if [ -z "$PREFIX" ]; then
    echo "monitor: could not auto-detect prefix from $LOG" >&2
    exit 1
  fi
fi

REPORT=~/PHC/logs/progress_report_${PREFIX}.md

echo "# Progress Report — Job ${JOBID} (${PREFIX})" > "$REPORT"
echo "_Started: $(date)_  log: \`${LOG}\`" >> "$REPORT"
echo "" >> "$REPORT"
echo "| Epoch | reward (rwd) | eps_len | time |" >> "$REPORT"
echo "|---|---|---|---|" >> "$REPORT"

LAST_MILESTONE=0
while true; do
  if ! squeue -j "$JOBID" -h 2>/dev/null | grep -q "$JOBID"; then
    echo "" >> "$REPORT"
    echo "_Job $JOBID no longer in queue at $(date)_" >> "$REPORT"
    break
  fi
  if [ -f "$LOG" ]; then
    LATEST=$(grep -oP "^${PREFIX}-Ep:\s*\K\d+" "$LOG" 2>/dev/null | tail -1)
    LATEST=${LATEST:-0}
    MILESTONE=$(( LATEST / 1000 * 1000 ))
    while [ "$MILESTONE" -gt "$LAST_MILESTONE" ]; do
      NEXT=$(( LAST_MILESTONE + 1000 ))
      LINE=$(grep -E "^${PREFIX}-Ep:\s*${NEXT}[[:space:]]" "$LOG" | tail -1)
      if [ -n "$LINE" ]; then
        RWD=$(echo "$LINE" | grep -oP 'rwd:\s*\K[0-9.\-]+')
        EPS_LEN=$(echo "$LINE" | grep -oP 'eps_len:\s*\K[0-9.\-]+')
        echo "| ${NEXT} | ${RWD:-?} | ${EPS_LEN:-?} | $(date '+%H:%M:%S') |" >> "$REPORT"
      fi
      LAST_MILESTONE=$NEXT
    done
  fi
  sleep 60
done
