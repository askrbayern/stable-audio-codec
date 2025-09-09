#!/bin/bash

set -euo pipefail

# ==============================================================================
# DNS clean_fullband extractor (only this folder)
# - Run via: sbatch download_data.sbatch /.../unzip_dns.sh
# - Parallel untar with progress
# ==============================================================================

: "${DNS_DIR:=/itet-stor/feigao/net_scratch/datasets/dns_challenge_4/clean_fullband}"
: "${PARALLEL_JOBS:=3}"

echo "===================================================="
echo "DNS Untar job starting"
echo "Host:        $(hostname)"
echo "Started at:  $(date)"
echo "DNS_DIR:     ${DNS_DIR}"
echo "PARALLEL:    ${PARALLEL_JOBS}"
echo "===================================================="

HAVE_PV=0
if command -v pv >/dev/null 2>&1; then
  HAVE_PV=1
  echo "pv found: will show byte progress"
else
  echo "pv not found: using tar checkpoints"
fi

if [[ -d "$DNS_DIR" ]]; then
  DNS_OUT="${DNS_DIR%/}/extracted"
  mkdir -p "$DNS_OUT"
  echo "[TBZ2] Scanning for tar.bz2 under $DNS_DIR ..."
  mapfile -d '' DNS_TARS < <(find "$DNS_DIR" -maxdepth 1 -type f -name "*.tar.bz2" -print0)
  echo "[TBZ2] Found ${#DNS_TARS[@]} archives"

  if [[ ${#DNS_TARS[@]} -eq 0 ]]; then
    echo "No archives to extract."
  else
    printf '%s\0' "${DNS_TARS[@]}" | xargs -0 -n 1 -P "${PARALLEL_JOBS}" -I{} bash -c '
set -euo pipefail
f="$1"; out="$2"; have_pv="$3";
echo "[TBZ2] -> $(basename -- "$f")";
if [ "$have_pv" = "1" ] && command -v pv >/dev/null 2>&1; then
  size=$(stat -c %s "$f" 2>/dev/null || stat -f %z "$f" 2>/dev/null || echo 0)
  if [ "$size" -gt 0 ]; then pv -s "$size" "$f" | tar -xj -C "$out" -f -; else pv "$f" | tar -xj -C "$out" -f -; fi
else
  tar -xjf "$f" -C "$out" --checkpoint=10000 --checkpoint-action=dot --totals; echo ""
fi
echo "[TBZ2] Done: $(basename -- "$f")";
' _ {} "$DNS_OUT" "$HAVE_PV"
  fi
else
  echo "[TBZ2] DNS_DIR not found at $DNS_DIR (skip)"
fi

echo "===================================================="
echo "Finished at: $(date)"
echo "===================================================="
