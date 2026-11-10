#!/usr/bin/env bash
# Use --cpu for installation and metadata checks on a machine without CUDA.
set -euo pipefail
source "$(dirname -- "$0")/common.sh"
MODE="${1:-cuda}"
if [[ "$MODE" != cuda && "$MODE" != --cpu ]]; then echo "Usage: $0 [--cpu]" >&2; exit 2; fi
"$PYTHON_BIN" - "$MODE" <<'PY'
import sys, torch
available = torch.cuda.is_available()
print(f"torch={torch.__version__} cuda_available={available} devices={torch.cuda.device_count()}")
if sys.argv[1] != '--cpu' and not available:
    raise SystemExit('CUDA is unavailable. Use the configured GPU environment or --cpu for setup checks.')
PY
"$PYTHON_BIN" -m pytest -q --disable-warnings
for DATASET in bdd100k nuimages; do
  "$PYTHON_BIN" -m revision.run audit-splits --dataset "$DATASET" --device cpu --output-dir "$OUTPUT_DIR"
done
"$PYTHON_BIN" - <<'PY'
from revision.data import disjoint_validation_rows, split_audit
from revision.paths import metadata_dir
rows, audit = disjoint_validation_rows('bdd100k', str(metadata_dir('bdd100k')))
assert len(rows) == 6986, 'BDD100K corrected validation must contain 6,986 images'
print(audit)
nu = split_audit('nuimages', str(metadata_dir('nuimages')))
assert nu['test_disjoint_from_train_and_validation']
print('Corrected BDD100K validation contains 6,986 images. nuImages splits are disjoint.')
PY
