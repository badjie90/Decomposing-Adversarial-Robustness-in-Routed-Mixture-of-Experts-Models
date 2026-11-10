#!/usr/bin/env bash
# Run inside tmux. Queues run sequentially; workers within a queue use separate GPUs.
set -euo pipefail
source "$(dirname -- "$0")/common.sh"
(( $# > 0 )) || { echo "Usage: $0 GPU_ID [GPU_ID ...]" >&2; exit 2; }
GPUS=("$@")
for GPU in "${GPUS[@]}"; do
  [[ "$GPU" =~ ^[0-9]+$ ]] || { echo 'GPU IDs must be nonnegative integers.' >&2; exit 2; }
done
PHASES=(primary controls transfer-scaling sparse-primary sparse-controls sensitivity)
MODELS=(csa csa csa sparse sparse csa)
for PHASE_INDEX in "${!PHASES[@]}"; do
  PHASE="${PHASES[$PHASE_INDEX]}"
  MODEL="${MODELS[$PHASE_INDEX]}"
  bash "$SCRIPT_DIR/04_generate_jobs.sh" "$PHASE" "$MODEL"
  JOBS="$OUTPUT_DIR/jobs/${PHASE}_${MODEL}.tsv"
  PIDS=()
  for INDEX in "${!GPUS[@]}"; do
    LOG="$OUTPUT_DIR/logs/queue_${PHASE}_${MODEL}_gpu${GPUS[$INDEX]}.log"
    bash "$SCRIPT_DIR/05_worker_entry.sh" "$JOBS" "$INDEX" "${#GPUS[@]}" "${GPUS[$INDEX]}" "$LOG" &
    PIDS+=("$!")
  done
  FAILED=0
  for PID in "${PIDS[@]}"; do wait "$PID" || FAILED=1; done
  if (( FAILED )); then echo "Queue failed $PHASE. Inspect results/logs and restart after resolving the error." >&2; exit 1; fi
  bash "$SCRIPT_DIR/07_audit_jobs.sh" "$JOBS"
done
echo 'All six queues passed their completion audits. Run scripts/06_report.sh to regenerate statistics.'
