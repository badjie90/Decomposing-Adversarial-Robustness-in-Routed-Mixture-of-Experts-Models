#!/usr/bin/env bash
# Run after the corrected BDD CSA checkpoint exists.  This selects exactly the
# same 1,000 IDs for every attack, seed, target and architecture within a dataset.
# Usage: bash scripts/03_make_manifests.sh 0 1
set -euo pipefail
source "$(dirname -- "$0")/common.sh"
BDD_GPU="${1:?Usage: $0 BDD_GPU NUIMAGES_GPU}"
NU_GPU="${2:?Usage: $0 BDD_GPU NUIMAGES_GPU}"

CUDA_VISIBLE_DEVICES="$BDD_GPU" "$PYTHON_BIN" -m revision.run make-manifest \
  --dataset bdd100k --metadata-dir "$BDD_METADATA" --model csa --run-dir "$BDD_CSA_RUN" \
  --device cuda:0 --n-images 1000 --seed 20261029 --stratification proportional \
  --output-dir "$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES="$NU_GPU" "$PYTHON_BIN" -m revision.run make-manifest \
  --dataset nuimages --metadata-dir "$NUIMAGES_METADATA" --model csa --run-dir "$NUIMAGES_CSA_RUN" \
  --device cuda:0 --n-images 1000 --seed 20261029 --stratification proportional \
  --output-dir "$OUTPUT_DIR"
