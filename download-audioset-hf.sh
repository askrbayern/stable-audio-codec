#!/usr/bin/env bash

set -euo pipefail

# ==============================================================================
# Download AudioSet from Hugging Face (agkphysics/AudioSet)
# Robust raw-file downloader: hf download + Python fallback, then extract tars
# ==============================================================================

# Configuration via environment variables
: "${DATASET_REPO:=agkphysics/AudioSet}"               # HF dataset repo id
: "${SPLIT:=all}"                                       # balanced | unbalanced | eval | all
: "${OUTPUT_DIR:=/itet-stor/feigao/net_scratch/datasets/audioset/hf}" # target root dir
: "${EXTRACT:=true}"                                    # true | false
: "${USE_FALLBACK:=true}"                               # use Python fallback if no tars fetched
: "${VENV_DIR:=}"                                       # optional venv path; default under OUTPUT_DIR

echo "===================================================="
echo "AudioSet HF raw downloader"
echo "Repo:        ${DATASET_REPO}"
echo "Split:       ${SPLIT}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Extract:     ${EXTRACT}"
echo "Started at:  $(date)"
echo "Host:        $(hostname)"
echo "===================================================="

mkdir -p "${OUTPUT_DIR}"

# Setup Python venv for huggingface_hub CLI/API
if [[ -z "${VENV_DIR}" ]]; then
  VENV_DIR="${OUTPUT_DIR}/.venv_hf"
fi
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating Python venv at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip >/dev/null
python -m pip install --upgrade "huggingface_hub[cli]>=0.23" >/dev/null

# Force HF caches to net_scratch (never use home)
CACHE_DIR="${OUTPUT_DIR}/.hf_cache"
mkdir -p "${CACHE_DIR}"
export HF_HOME="${CACHE_DIR}"
export HUGGINGFACE_HUB_CACHE="${CACHE_DIR}"
export TRANSFORMERS_CACHE="${CACHE_DIR}"
export HF_HUB_ENABLE_HF_TRANSFER=1

# Optional: pick up HF token if provided
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"

DOWNLOAD_DIR="${OUTPUT_DIR}/hf_repo"
mkdir -p "${DOWNLOAD_DIR}"
export DOWNLOAD_DIR DATASET_REPO SPLIT CACHE_DIR

# Resolve hf CLI within this venv to avoid wrong interpreter issues
HF_CMD="${VENV_DIR}/bin/hf"
if [[ ! -x "${HF_CMD}" ]]; then
  HF_CMD="python -m huggingface_hub.cli.hf"
fi

# Build include patterns based on SPLIT
declare -a INCLUDE_ARGS
INCLUDE_ARGS+=(--include "data/ontology.json")
case "${SPLIT}" in
  balanced)
    INCLUDE_ARGS+=(--include "data/bal_train*.tar" --include "data/balanced_train_segments.csv")
    ;;
  unbalanced)
    INCLUDE_ARGS+=(--include "data/unbal_train*.tar")
    ;;
  eval)
    INCLUDE_ARGS+=(--include "data/eval*.tar" --include "data/eval_segments.csv")
    ;;
  all)
    INCLUDE_ARGS+=(--include "data/bal_train*.tar" --include "data/unbal_train*.tar" --include "data/eval*.tar" --include "data/*segments.csv")
    ;;
  *)
    echo "Unknown SPLIT='${SPLIT}'. Use balanced|unbalanced|eval|all" >&2
    exit 1
    ;;
esac

echo "Downloading selected files from ${DATASET_REPO} via hf download ..."
set +e
${HF_CMD} download "${DATASET_REPO}" \
  --repo-type dataset \
  --local-dir "${DOWNLOAD_DIR}" \
  "${INCLUDE_ARGS[@]}"
hf_status=$?
set -e

num_tars=$(find "${DOWNLOAD_DIR}/data" -type f -name "*.tar" 2>/dev/null | wc -l | tr -d ' ')
echo "hf download exit code: ${hf_status}; tar archives found: ${num_tars}"

# Fallback: use huggingface_hub Python API to enumerate and fetch matching files
if [[ "${USE_FALLBACK}" == "true" && "${num_tars}" == "0" ]]; then
  echo "No tar archives fetched by hf download. Falling back to Python API downloader..."
  # Heartbeat monitor prints counts and size every 60s
  PROGRESS_DIR="${DOWNLOAD_DIR}/data"
  (
    while true; do
      ts=$(date +%H:%M:%S)
      count=$(find "${PROGRESS_DIR}" -type f -name "*.tar" 2>/dev/null | wc -l | tr -d ' ')
      size=$(du -sh "${PROGRESS_DIR}" 2>/dev/null | awk '{print $1}')
      echo "[${ts}] Heartbeat: tar=${count}, size=${size}"
      sleep 60
    done
  ) &
  HEARTBEAT_PID=$!
  python - <<'PY'
import fnmatch
import os
from pathlib import Path
from typing import List
from huggingface_hub import HfApi, hf_hub_download

repo_id = os.environ.get("DATASET_REPO", "agkphysics/AudioSet")
download_root = Path(os.environ["DOWNLOAD_DIR"])  # points to .../hf_repo
download_root.mkdir(parents=True, exist_ok=True)
cache_dir = os.environ.get("CACHE_DIR")

split = os.environ.get("SPLIT", "all")
patterns: List[str] = ["data/ontology.json"]
if split in ("balanced", "all"):
    patterns.append("data/bal_train*.tar")
    patterns.append("data/balanced_train_segments.csv")
if split in ("unbalanced", "all"):
    patterns.append("data/unbal_train*.tar")
if split in ("eval", "all"):
    patterns.append("data/eval*.tar")
    patterns.append("data/eval_segments.csv")

def wanted(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pat) for pat in patterns)

api = HfApi()
files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
matched = [f for f in files if wanted(f)]
matched.sort()
total = len(matched)
num_tars = sum(1 for f in matched if f.endswith('.tar'))
num_csv = sum(1 for f in matched if f.endswith('.csv'))
print(f"Matched {total} files via API fallback (tars={num_tars}, csv={num_csv})")

for idx, rel_path in enumerate(matched, start=1):
    target_path = download_root / rel_path  # preserve repo-relative path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        print(f"[{idx}/{total}] Skip existing: {target_path}")
        continue
    last_err = None
    for attempt in range(1, 6):
        try:
            print(f"[{idx}/{total}] Downloading: {rel_path} (attempt {attempt})")
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=rel_path,
                repo_type="dataset",
                cache_dir=cache_dir,
                local_dir=str(download_root),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            # Ensure file is at expected target path
            if Path(local_path) != target_path:
                # In rare cases, hf may return a cache path; copy to target
                import shutil
                shutil.move(local_path, target_path)
            size_mb = target_path.stat().st_size / (1024 * 1024)
            print(f"[{idx}/{total}] Done: {target_path.name} ({size_mb:.1f} MB)")
            break
        except Exception as e:
            last_err = e
            print(f"[{idx}/{total}] Error: {e}; retrying in 10s ...")
            import time; time.sleep(10)
    else:
        print(f"[{idx}/{total}] Failed permanently: {rel_path} -> {last_err}")
PY
  # Stop heartbeat monitor
  if [[ -n "${HEARTBEAT_PID:-}" ]]; then
    kill "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
  fi
  num_tars=$(find "${DOWNLOAD_DIR}/data" -type f -name "*.tar" 2>/dev/null | wc -l | tr -d ' ')
fi

echo "Total tar archives ready: ${num_tars}"

if [[ "${EXTRACT}" == "true" ]]; then
  EXTRACT_DIR="${OUTPUT_DIR}/extracted"
  mkdir -p "${EXTRACT_DIR}"
  echo "Extracting tar archives into ${EXTRACT_DIR} ..."
  while IFS= read -r -d '' tarfile; do
    base=$(basename "${tarfile}")
    echo "- Extracting ${base}"
    tar -xf "${tarfile}" -C "${EXTRACT_DIR}"
  done < <(find "${DOWNLOAD_DIR}/data" -type f -name "*.tar" -print0)

  flac_count=$(find "${EXTRACT_DIR}" -type f -name "*.flac" 2>/dev/null | wc -l | tr -d ' ')
  wav_count=$(find "${EXTRACT_DIR}" -type f -name "*.wav" 2>/dev/null | wc -l | tr -d ' ')
  echo "Extraction complete. Found ${flac_count} .flac and ${wav_count} .wav files."
else
  echo "Skipping extraction as EXTRACT=${EXTRACT}"
fi

echo "===================================================="
echo "Finished at: $(date)"
echo "Output root: ${OUTPUT_DIR}"
echo "HF repo dir: ${DOWNLOAD_DIR}"
if [[ "${EXTRACT}" == "true" ]]; then
  echo "Extracted to: ${OUTPUT_DIR}/extracted"
fi
echo "===================================================="