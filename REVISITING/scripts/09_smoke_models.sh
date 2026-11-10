#!/usr/bin/env bash
# Bounded one-image checks. This script does not launch the experimental queues.
set -euo pipefail
source "$(dirname -- "$0")/common.sh"
DEVICE="${1:-cpu}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
for DATASET in bdd100k nuimages; do
  "$PYTHON_BIN" -m revision.run smoke --dataset "$DATASET" --model csa --device "$DEVICE" --smoke-queries 4
  "$PYTHON_BIN" -m revision.run smoke --dataset "$DATASET" --model sparse --device "$DEVICE" \
    --sparse-checkpoint "$OUTPUT_DIR/models/$DATASET/sparse_seed42/checkpoints/best.pt" --smoke-queries 4
done
