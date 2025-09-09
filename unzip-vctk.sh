#!/bin/bash

set -euo pipefail

# ==============================================================================
# VCTK unzipper (only this file)
# - Run via: sbatch download_data.sbatch /.../unzip_vctk.sh
# ==============================================================================

: "${VCTK_ZIP:=/itet-stor/feigao/net_scratch/datasets/vctk/DS_10283_3443.zip}"
VCTK_DIR="${VCTK_ZIP%.zip}"
# Allow overriding the inner zip path; default to the common name inside the outer folder
: "${VCTK_INNER_ZIP:=${VCTK_DIR}/VCTK-Corpus-0.92.zip}"

echo "===================================================="
echo "VCTK Unzip job starting"
echo "Host:        $(hostname)"
echo "Started at:  $(date)"
echo "VCTK_ZIP:    ${VCTK_ZIP}"
echo "INNER_ZIP:   ${VCTK_INNER_ZIP}"
echo "===================================================="

if [[ -f "$VCTK_ZIP" ]]; then
  mkdir -p "$VCTK_DIR"
  echo "[ZIP] Extracting $(basename "$VCTK_ZIP") -> $VCTK_DIR"
  unzip -o "$VCTK_ZIP" -d "$VCTK_DIR"
  echo "[ZIP] Done: $(basename "$VCTK_ZIP")"
else
  echo "[ZIP] VCTK outer zip not found at $VCTK_ZIP (skip outer)"
fi

# Now extract the inner corpus zip if present
if [[ -f "$VCTK_INNER_ZIP" ]]; then
  echo "[ZIP] Extracting inner $(basename "$VCTK_INNER_ZIP") -> $VCTK_DIR"
  unzip -o "$VCTK_INNER_ZIP" -d "$VCTK_DIR"
  echo "[ZIP] Done: $(basename "$VCTK_INNER_ZIP")"
else
  echo "[ZIP] Inner VCTK zip not found at $VCTK_INNER_ZIP (skip inner)"
fi

echo "===================================================="
echo "Finished at: $(date)"
echo "===================================================="

