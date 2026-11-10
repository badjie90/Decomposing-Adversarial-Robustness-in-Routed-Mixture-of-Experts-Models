#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/common.sh"
"$PYTHON_BIN" -m revision.report --output-dir "$OUTPUT_DIR" --bootstrap-replicates 5000 --bootstrap-seed 20261029
