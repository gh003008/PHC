#!/bin/bash
# Poll Slurm log for PHC training progress and report every 1000 epochs.
# Log format:  <exp>-Ep: <N>	rwd: <R>	fps_step: ...	eps_len: <L>
# Usage: ./monitor_progress.sh <jobid>  (default: latest)

JOBID="${1:-}"
if [ -z "$JOBID" ]; then
  JOBID=$(ls -t ~/PHC/logs/phc_walk_server_*.out 2>/dev/null | head -1 | grep -oP '\d+(?=\.out$)')
fi
LOG=~/PHC/logs/phc_walk_server_${JOBID}.out
REPORT=~/PHC/logs/progress_report.md

echo "# Progress Report — Job ${JOBID}" > "$REPORT"
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
    LATEST=$(grep -oP 'Ep:\s*\K\d+' "$LOG" 2>/dev/null | tail -1)
    LATEST=${LATEST:-0}
    MILESTONE=$(( LATEST / 1000 * 1000 ))
    while [ "$MILESTONE" -gt "$LAST_MILESTONE" ]; do
      NEXT=$(( LAST_MILESTONE + 1000 ))
      LINE=$(grep -E "Ep:\s*${NEXT}[[:space:]]" "$LOG" | tail -1)
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
