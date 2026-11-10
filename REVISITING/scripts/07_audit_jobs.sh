#!/usr/bin/env bash
# Verify that every row in a queue has a readable, identity-matched artifact family.
set -euo pipefail
source "$(dirname -- "$0")/common.sh"

JOBS="${1:?Usage: $0 results/jobs/PHASE.tsv}"
"$PYTHON_BIN" - "$JOBS" "$OUTPUT_DIR" <<'PY'
import csv
import sys
from pathlib import Path

from revision.jobs import job_is_complete

jobs_file = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
rows = list(csv.DictReader(jobs_file.open(encoding="utf-8"), delimiter="\t"))
incomplete = [row for row in rows if not job_is_complete(row, output_dir, (100, 250, 500, 1000))]
print(f"{jobs_file.name}: {len(rows) - len(incomplete)}/{len(rows)} complete")
if incomplete:
    print("Incomplete queue rows:")
    for row in incomplete:
        print("  " + " ".join(f"{key}={value}" for key, value in row.items()))
    raise SystemExit(1)
PY
