#!/bin/bash

#SBATCH --job-name=rsync_audio_to_scratch
#SBATCH --output=/itet-stor/feigao/net_scratch/logs/rsync_audio_%j.out
#SBATCH --error=/itet-stor/feigao/net_scratch/logs/rsync_audio_%j.err
#SBATCH --nodelist=tikgpu10
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=12:00:00

set -euo pipefail

# ===================== User settings (edit these two) =====================
# Source directory (already-extracted audio on net_scratch)
SRC_DIR="/itet-stor/feigao/net_scratch/datasets/dns_challenge_4/clean_fullband/extracted/datasets_fullband/clean_fullband"

# Target directory name under DST_BASE (no trailing slash)
# e.g., "dataset/dns_challenge_4_clean" -> /scratch/$USER/dataset/dns_challenge_4_clean
TARGET_NAME="dataset/dns_challenge_4_clean"

# Optional: change destination base to nest under custom path
# e.g., DST_BASE="/scratch/$USER/feigao/data"
DST_BASE="/scratch/$USER"/datasets
# ========================================================================

DST_DIR="${DST_BASE%/}/${TARGET_NAME}"

echo "===================================================="
echo "rsync_audio_to_scratch"
echo "Host:        $(hostname)"
echo "Started at:  $(date)"
echo "SRC_DIR:     ${SRC_DIR}"
echo "DST_DIR:     ${DST_DIR}"
echo "===================================================="

if [[ ! -d "$SRC_DIR" ]]; then
  echo "ERROR: SRC_DIR not found: $SRC_DIR" >&2
  exit 1
fi

mkdir -p "$DST_DIR"

if [[ "$DST_DIR" == "$SRC_DIR"* ]]; then
  echo "ERROR: DST_DIR appears inside SRC_DIR; pick a different TARGET_NAME/DST_BASE" >&2
  exit 1
fi

echo "Stage 1/2: rsync audio (ignore existing, no overwrite)"
ulimit -n 524288 || true

# Lower CPU/IO priority to reduce contention with training
IONICE="ionice -c2 -n7"; NICE="nice -n 10"

# Only copy common audio files; never overwrite existing; avoid showing half files to readers
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"
if [[ "$SHOW_PROGRESS" = "1" ]]; then
  RSYNC_PROGRESS_OPT="--info=progress2"
else
  RSYNC_PROGRESS_OPT=""
fi

$NICE $IONICE rsync -aH \
  $RSYNC_PROGRESS_OPT --partial --partial-dir=.rsync-partial \
  --ignore-existing --prune-empty-dirs \
  --include="*/" \
  --include="*.wav" --include="*.flac" --include="*.mp3" --include="*.ogg" --include="*.opus" --include="*.m4a" \
  --include="*.w64" --include="*.aif" --include="*.aiff" --include="*.caf" \
  --exclude="*" \
  "${SRC_DIR%/}/" "${DST_DIR%/}/"

echo "Stage 2/2: generate filelist.txt"
out_file="$DST_DIR/filelist.txt"
tmp_file="$out_file.tmp"

find "$DST_DIR" -type f \
  \( -iname '*.wav' -o -iname '*.flac' -o -iname '*.ogg' -o -iname '*.opus' -o -iname '*.m4a' -o -iname '*.mp3' -o -iname '*.w64' -o -iname '*.aif' -o -iname '*.aiff' -o -iname '*.caf' \) \
  -not -name '._*' -not -path '*/__MACOSX/*' \
  -printf '%P\n' | LC_ALL=C sort > "$tmp_file"

mv -f "$tmp_file" "$out_file"
echo "[OK] $out_file ($(wc -l < "$out_file") files)"

echo "Finished at: $(date)"
echo "===================================================="

