#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export REVISION_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
# Explicit environment overrides take precedence over local configuration.
# Do not retain unexported model defaults from an earlier source of this file.
_moe_load_paths() {
  local name declaration previous_flags="$-"
  local -A caller=()
  local -a settings=(OUTPUT_DIR PYTHON_BIN BDD_METADATA NUIMAGES_METADATA
    BDD_IMAGE_ROOT NUIMAGES_IMAGE_ROOT BDD_CSA_RUN NUIMAGES_CSA_RUN MPLCONFIGDIR)
  for name in "${settings[@]}"; do
    declaration="$(declare -p "$name" 2>/dev/null || true)"
    if [[ "$declaration" =~ ^declare\ -[^[:space:]]*x ]]; then
      caller["$name"]="${!name}"
    fi
  done
  unset BDD_CSA_RUN NUIMAGES_CSA_RUN
  if [[ -f "$REVISION_ROOT/configs/paths.env" ]]; then
    set -a
    source "$REVISION_ROOT/configs/paths.env"
    if [[ "$previous_flags" != *a* ]]; then set +a; fi
  fi
  for name in "${!caller[@]}"; do
    printf -v "$name" '%s' "${caller[$name]}"
    export "$name"
  done
}
_moe_load_paths
unset -f _moe_load_paths
export OUTPUT_DIR="${OUTPUT_DIR:-$REVISION_ROOT/results}"
if [[ "$OUTPUT_DIR" != /* ]]; then OUTPUT_DIR="$REVISION_ROOT/$OUTPUT_DIR"; fi
export PYTHON_BIN="$(command -v "${PYTHON_BIN:-python}")"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REVISION_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REVISION_ROOT/.mplconfig}"
export BDD_METADATA="${BDD_METADATA:-$REVISION_ROOT/data/metadata/bdd100k}"
export NUIMAGES_METADATA="${NUIMAGES_METADATA:-$REVISION_ROOT/data/metadata/nuimages}"
export BDD_IMAGE_ROOT="${BDD_IMAGE_ROOT:-$REVISION_ROOT/data/bdd100k}"
export NUIMAGES_IMAGE_ROOT="${NUIMAGES_IMAGE_ROOT:-$REVISION_ROOT/data/nuimages}"
BDD_CSA_RUN="${BDD_CSA_RUN:-$OUTPUT_DIR/models/bdd100k/csa_disjoint_seed42}"
NUIMAGES_CSA_RUN="${NUIMAGES_CSA_RUN:-$OUTPUT_DIR/models/nuimages/csa_seed42}"
mkdir -p "$OUTPUT_DIR/logs" "$MPLCONFIGDIR"
cd "$REVISION_ROOT"
