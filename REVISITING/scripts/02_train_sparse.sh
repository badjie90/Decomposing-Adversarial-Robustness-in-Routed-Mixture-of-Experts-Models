#!/usr/bin/env bash
# Usage: bash scripts/02_train_sparse.sh {bdd100k|nuimages} GPU_ID
set -euo pipefail
source "$(dirname -- "$0")/common.sh"
# Do not put a literal `}` in a `${parameter:?message}` expansion: Bash would
# treat it as the end of the expansion and append the rest to the argument.
DATASET="${1:?Usage: $0 DATASET GPU_ID}"
GPU="${2:?Usage: $0 DATASET GPU_ID}"
case "$DATASET" in
  bdd100k) METADATA="$BDD_METADATA" ;;
  nuimages) METADATA="$NUIMAGES_METADATA" ;;
  *) echo "Unknown dataset: $DATASET" >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES="$GPU"
"$PYTHON_BIN" -m revision.train_sparse --dataset "$DATASET" --metadata-dir "$METADATA" \
  --output-dir "$OUTPUT_DIR/models/$DATASET/sparse_seed42" --device cuda:0 --seed 42 \
  --epochs 50 --batch-size 64 --num-workers 8 --early-stopping-patience 10
