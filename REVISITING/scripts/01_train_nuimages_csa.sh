#!/usr/bin/env bash
# Usage: bash scripts/01_train_nuimages_csa.sh GPU_ID
set -euo pipefail
source "$(dirname -- "$0")/common.sh"
GPU="${1:?Usage: $0 GPU_ID}"
export CUDA_VISIBLE_DEVICES="$GPU"
NUIMAGES_CSA_RUN="${NUIMAGES_CSA_RUN:-$OUTPUT_DIR/models/nuimages/csa_seed42}"

# Verify the prepared split before starting the preserved training routine.
"$PYTHON_BIN" - "$NUIMAGES_METADATA" "$NUIMAGES_CSA_RUN" <<'PY'
import sys
from pathlib import Path

import torch
from revision.data import split_audit

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Activate the configured GPU environment.")
audit = split_audit("nuimages", sys.argv[1])
if any(item["count"] for item in audit["overlap"].values()):
    raise SystemExit("nuImages training requires disjoint train, validation and fixed-test splits.")
output = Path(sys.argv[2])
if output.exists() and any(output.iterdir()):
    raise SystemExit(f"Output directory is not empty: {output}. Use a new run directory.")
print(audit, flush=True)
PY

"$PYTHON_BIN" "$REVISION_ROOT/baseline/nuimages_moe_train.py" train \
  --metadata-dir "$NUIMAGES_METADATA" --experiment-dir "$NUIMAGES_CSA_RUN" \
  --stage stage3_moe --backbone-name convnextv2_tiny.fcmae_ft_in1k \
  --image-size 224 --batch-size 32 --epochs 50 --num-workers 8 --seed 42 \
  --amp-dtype float16 --backbone-lr 2e-5 --expert-lr 1e-4 --head-lr 1e-4 \
  --weight-decay 1e-4 --min-lr 1e-6 --grad-clip-norm 1 \
  --early-stopping-patience 10 --early-stopping-min-delta 1e-4
