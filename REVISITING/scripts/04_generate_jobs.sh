#!/usr/bin/env bash
# Usage examples:
#   bash scripts/04_generate_jobs.sh primary csa
#   bash scripts/04_generate_jobs.sh controls csa
#   bash scripts/04_generate_jobs.sh transfer-scaling csa
#   bash scripts/04_generate_jobs.sh sparse-primary sparse
set -euo pipefail
source "$(dirname -- "$0")/common.sh"
PHASE="${1:?Usage: $0 PHASE MODEL [DATASET ...]}"
MODEL="${2:?Usage: $0 PHASE MODEL [DATASET ...]}"
shift 2
DATASETS=("$@")
if (( ${#DATASETS[@]} == 0 )); then DATASETS=(bdd100k nuimages); fi
JOBS="$OUTPUT_DIR/jobs/${PHASE}_${MODEL}.tsv"
mkdir -p "$(dirname -- "$JOBS")"
"$PYTHON_BIN" -m revision.jobs generate --phase "$PHASE" --models "$MODEL" --datasets "${DATASETS[@]}" --jobs-file "$JOBS"
echo "$JOBS"
