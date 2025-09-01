#!/usr/bin/env bash
set -euo pipefail

# Config path: hardcoded default; can be overridden by arg1 or CFG_JSON env
CFG_DEFAULT="/itet-stor/feigao/home/stable-audio-tools/stable_audio_tools/configs/dataset_configs/test_training_partial.json"
CFG="${1:-${CFG_JSON:-$CFG_DEFAULT}}"

# sbatch --export=ALL,CFG_JSON=/itet-stor/feigao/home/stable-audio-tools/stable_audio_tools/configs/dataset_configs/test_training.json \
#   /itet-stor/feigao/home/stable-audio-tools/process_data.sbatch \
#   /itet-stor/feigao/home/stable-audio-tools/generate_filelists.sh

if [ ! -f "$CFG" ]; then
  echo "Config not found: $CFG" >&2
  exit 1
fi

# Extract all dataset paths from JSON (jq preferred; robust Python fallback)
if command -v jq >/dev/null 2>&1; then
  mapfile -t PATHS < <(jq -r '.datasets[]?.path' "$CFG")
else
  echo "jq not found; using Python fallback" >&2
  mapfile -t PATHS < <(python3 - "$CFG" <<'PY'
import json, sys
cfg = sys.argv[1]
with open(cfg, 'r') as f:
    data = json.load(f)
for ds in data.get('datasets', []):
    p = ds.get('path')
    if isinstance(p, str):
        print(p)
PY
)
fi

if [ ${#PATHS[@]} -eq 0 ]; then
  echo "No dataset paths found in $CFG" >&2
  exit 1
fi

echo "Generating filelist.txt for ${#PATHS[@]} roots..."

for ROOT in "${PATHS[@]}"; do
  if [ ! -d "$ROOT" ]; then
    echo "[SKIP] Not a directory: $ROOT" >&2
    continue
  fi
  echo "[ROOT] $ROOT"
  out="$ROOT/filelist.txt"

  find "$ROOT" -type f \
    \( -iname '*.wav' -o -iname '*.flac' -o -iname '*.ogg' -o -iname '*.opus' -o -iname '*.m4a' -o -iname '*.mp3' \) \
    -not -name '._*' -not -path '*/__MACOSX/*' \
    -printf '%P\n' \
    | LC_ALL=C sort > "$out.tmp"

  mv -f "$out.tmp" "$out"
  echo "[OK] $out ($(wc -l < "$out") files)"
done

echo "Done."
