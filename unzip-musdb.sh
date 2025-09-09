#!/bin/bash

set -euo pipefail

# ==============================================================================
# MUSDB unzipper (only this file)
# - Run via: sbatch download_data.sbatch /.../unzip_musdb.sh
# ==============================================================================

: "${MUSDB_ZIP:=/itet-stor/feigao/net_scratch/datasets/musdb/musdb18hq.zip}"

echo "===================================================="
echo "MUSDB Unzip job starting"
echo "Host:        $(hostname)"
echo "Started at:  $(date)"
echo "MUSDB_ZIP:   ${MUSDB_ZIP}"
echo "===================================================="

if [[ -f "$MUSDB_ZIP" ]]; then
  MUSDB_DIR="${MUSDB_ZIP%.zip}"
  mkdir -p "$MUSDB_DIR"
  echo "[ZIP] Extracting $(basename "$MUSDB_ZIP") -> $MUSDB_DIR"
  unzip -o "$MUSDB_ZIP" -d "$MUSDB_DIR"
  echo "[ZIP] Done: $(basename "$MUSDB_ZIP")"
else
  echo "[ZIP] MUSDB not found at $MUSDB_ZIP (skip)"
fi

echo "===================================================="
echo "Finished at: $(date)"
echo "===================================================="

