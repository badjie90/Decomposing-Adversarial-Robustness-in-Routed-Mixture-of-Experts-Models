#!/usr/bin/env bash
# One detached tmux worker per GPU. See 08_run_all_queues.sh for automatic phases.
set -euo pipefail
source "$(dirname -- "$0")/common.sh"
JOBS="${1:?Usage: $0 JOBS_FILE [--dry-run] GPU_ID [GPU_ID ...]}"
shift
if [[ "$JOBS" != /* ]]; then JOBS="$REVISION_ROOT/$JOBS"; fi
DRY=0
if [[ "${1:-}" == --dry-run ]]; then DRY=1; shift; fi
(( $# > 0 )) || { echo 'Supply at least one GPU ID.' >&2; exit 2; }
[[ -f "$JOBS" ]] || { echo "Missing queue $JOBS" >&2; exit 2; }
command -v tmux >/dev/null || { echo 'tmux is required.' >&2; exit 2; }
GPUS=("$@")
PREFIX="${SESSION_PREFIX:-moe}"
for GPU in "${GPUS[@]}"; do
  [[ "$GPU" =~ ^[0-9]+$ ]] || { echo 'GPU IDs must be nonnegative integers.' >&2; exit 2; }
  SESSION="${PREFIX}_$(basename -- "$JOBS" .tsv)_gpu${GPU}"
  if tmux has-session -t "=$SESSION" 2>/dev/null; then echo "Session already exists $SESSION" >&2; exit 2; fi
done
for INDEX in "${!GPUS[@]}"; do
  GPU="${GPUS[$INDEX]}"
  SESSION="${PREFIX}_$(basename -- "$JOBS" .tsv)_gpu${GPU}"
  LOG="$OUTPUT_DIR/logs/$SESSION.log"
  printf -v COMMAND '%q ' env "OUTPUT_DIR=$OUTPUT_DIR" "PYTHON_BIN=$PYTHON_BIN" \
    "BDD_METADATA=$BDD_METADATA" "NUIMAGES_METADATA=$NUIMAGES_METADATA" \
    "BDD_IMAGE_ROOT=$BDD_IMAGE_ROOT" "NUIMAGES_IMAGE_ROOT=$NUIMAGES_IMAGE_ROOT" \
    "BDD_CSA_RUN=$BDD_CSA_RUN" "NUIMAGES_CSA_RUN=$NUIMAGES_CSA_RUN" \
    "WORKER_DRY_RUN=$DRY" bash "$SCRIPT_DIR/05_worker_entry.sh" "$JOBS" \
    "$INDEX" "${#GPUS[@]}" "$GPU" "$LOG"
  tmux new-session -d -s "$SESSION" "$COMMAND"
  echo "Started $SESSION. Log $LOG"
done
