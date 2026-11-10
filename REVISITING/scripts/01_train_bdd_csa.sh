#!/usr/bin/env bash
# Usage: bash scripts/01_train_bdd_csa.sh 0
set -euo pipefail
source "$(dirname -- "$0")/common.sh"
GPU="${1:?Usage: $0 GPU_ID}"
export CUDA_VISIBLE_DEVICES="$GPU"

"$PYTHON_BIN" -m revision.train_csa \
  --metadata-dir "$BDD_METADATA" \
  --output-dir "$BDD_CSA_RUN" \
  --device cuda:0 --seed 42 --epochs 100 --batch-size 32 --num-workers 8 \
  --early-stopping-patience 10 --amp-dtype float16
