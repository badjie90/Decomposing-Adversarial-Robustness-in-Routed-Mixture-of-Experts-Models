#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/common.sh"
JOBS="$1"
INDEX="$2"
WORKERS="$3"
GPU="$4"
LOG="$5"
EXTRA=()
if [[ "${WORKER_DRY_RUN:-0}" == 1 ]]; then EXTRA+=(--dry-run); fi
STATUS=0
"$PYTHON_BIN" -m revision.jobs worker --jobs-file "$JOBS" --output-dir "$OUTPUT_DIR" \
  --worker-index "$INDEX" --workers "$WORKERS" --gpu "$GPU" "${EXTRA[@]}" \
  > "$LOG" 2>&1 || STATUS=$?
printf '%s\n' "$STATUS" > "${LOG%.log}.exitcode"
exit "$STATUS"
