#!/usr/bin/bash

# ==============================================================================
# AudioSet Unbalanced Train Segments Downloader
# ==============================================================================

# Configuration
CSV_FILE="unbalanced_train_segments.csv"
OUTPUT_DIR="/itet-stor/feigao/net_scratch/datasets/audioset/unbalanced_train_segments"
TEMP_DIR="./audioset_temp_unbalanced"

# Check dependencies
missing_deps=()
for cmd in yt-dlp ffmpeg bc; do
    if ! command -v "$cmd" &> /dev/null; then
        missing_deps+=("$cmd")
    fi
done

if [ ${#missing_deps[@]} -gt 0 ]; then
    echo "Error: Missing dependencies: ${missing_deps[*]}"
    echo "Please install them before running this script."
    exit 1
fi

# Check if CSV file exists
if [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file '$CSV_FILE' not found."
    exit 1
fi

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEMP_DIR"

# Initialize counters
SUCCESS_COUNT=0
FAILURE_COUNT=0
SKIPPED_COUNT=0
PROCESSED_COUNT=0

# Calculate total lines (skip first 3 comment lines)
TOTAL_LINES=$(tail -n +4 "$CSV_FILE" | wc -l)

echo "Script started at: $(date)"
echo "Processing: $CSV_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Total $TOTAL_LINES entries to process."
echo "===================================================="

# Function to process a single video
process_video() {
    local ytid="$1"
    local start="$2"
    local end="$3"
    local output="$4"
    
    local temp_audio="${TEMP_DIR}/${ytid}.wav"
    
    # Download full audio with improved options
    if ! yt-dlp --quiet --extract-audio --audio-format wav \
        --user-agent "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" \
        --sleep-interval 2 --max-sleep-interval 5 \
        --socket-timeout 30 --no-check-certificate \
        --retries 3 --fragment-retries 3 \
        -o "$temp_audio" "https://www.youtube.com/watch?v=$ytid" 2>/dev/null; then
        return 1
    fi
    
    # Check if downloaded file exists
    if [ ! -f "$temp_audio" ]; then
        return 1
    fi
    
    # Clip audio segment
    if ! ffmpeg -hide_banner -loglevel error \
        -i "$temp_audio" -ss "$start" -to "$end" \
        -c copy "$output" 2>/dev/null; then
        rm -f "$temp_audio"
        return 1
    fi
    
    # Clean up temporary file
    rm -f "$temp_audio"
    return 0
}

# Use temporary file to avoid subshell issues
TEMP_CSV=$(mktemp)
tail -n +4 "$CSV_FILE" > "$TEMP_CSV"

# Process each line in the CSV
while IFS=, read -r YTID START_SECONDS END_SECONDS LABELS || [ -n "$YTID" ]; do
    
    # Skip empty lines
    [ -z "$YTID" ] && continue
    
    # Increment processed counter
    ((PROCESSED_COUNT++))

    # Clean whitespace and quotes
    YTID=$(echo "$YTID" | xargs | sed 's/^"//;s/"$//')
    START_SECONDS=$(echo "$START_SECONDS" | xargs | sed 's/^"//;s/"$//')
    END_SECONDS=$(echo "$END_SECONDS" | xargs | sed 's/^"//;s/"$//')

    # Validate required fields
    if [ -z "$YTID" ] || [ -z "$START_SECONDS" ] || [ -z "$END_SECONDS" ]; then
        echo "[$PROCESSED_COUNT/$TOTAL_LINES] Skip: Missing required fields (YTID: $YTID, START: $START_SECONDS, END: $END_SECONDS)"
        ((FAILURE_COUNT++))
        continue
    fi

    # Validate time format (should be numeric)
    if ! [[ "$START_SECONDS" =~ ^[0-9]+\.?[0-9]*$ ]] || ! [[ "$END_SECONDS" =~ ^[0-9]+\.?[0-9]*$ ]]; then
        echo "[$PROCESSED_COUNT/$TOTAL_LINES] Skip: Invalid time format (START: $START_SECONDS, END: $END_SECONDS)"
        ((FAILURE_COUNT++))
        continue
    fi

    # Validate time logic (end time should be greater than start time)
    if (( $(echo "$END_SECONDS <= $START_SECONDS" | bc -l) )); then
        echo "[$PROCESSED_COUNT/$TOTAL_LINES] Skip: Invalid time range (START: $START_SECONDS >= END: $END_SECONDS)"
        ((FAILURE_COUNT++))
        continue
    fi

    # Define output filename and path
    FINAL_FILENAME="${YTID}_${START_SECONDS}_${END_SECONDS}.wav"
    FINAL_FILEPATH="${OUTPUT_DIR}/${FINAL_FILENAME}"

    # Check if file already exists (resume capability)
    if [ -f "$FINAL_FILEPATH" ]; then
        echo "[$PROCESSED_COUNT/$TOTAL_LINES] Skip: $FINAL_FILENAME (already exists)"
        ((SKIPPED_COUNT++))
        continue
    fi

    echo "[$PROCESSED_COUNT/$TOTAL_LINES] Processing: $YTID, time: ${START_SECONDS}s - ${END_SECONDS}s"

    # Process the video
    if process_video "$YTID" "$START_SECONDS" "$END_SECONDS" "$FINAL_FILEPATH"; then
        echo "[$PROCESSED_COUNT/$TOTAL_LINES] Success: Created $FINAL_FILENAME"
        ((SUCCESS_COUNT++))
    else
        echo "[$PROCESSED_COUNT/$TOTAL_LINES] Failed: Cannot process $YTID (network issue or video unavailable)"
        # Log failed video ID for later retry
        echo "$YTID,$START_SECONDS,$END_SECONDS" >> "${OUTPUT_DIR}/failed_videos.csv"
        ((FAILURE_COUNT++))
        ((GLOBAL_FAILURE_COUNT++))
        # Clean up incomplete files
        rm -f "$FINAL_FILEPATH"
    fi

done < "$TEMP_CSV"

# Clean up temporary file
rm -f "$TEMP_CSV"

echo "===================================================="
echo "Script completed at: $(date)"
echo ""
echo "FINAL REPORT"
echo "------------------------------------"
echo "Success: $SUCCESS_COUNT"
echo "Failed: $FAILURE_COUNT"
echo "Skipped (already exists): $SKIPPED_COUNT"
echo "Total processed: $PROCESSED_COUNT / $TOTAL_LINES"
echo "------------------------------------"

# Calculate success rate
if [ $TOTAL_LINES -gt 0 ]; then
    SUCCESS_RATE=$(echo "scale=2; $SUCCESS_COUNT * 100 / $TOTAL_LINES" | bc -l)
    echo "Success rate: ${SUCCESS_RATE}%"
fi

echo "Output directory: $OUTPUT_DIR"
if [ -d "$OUTPUT_DIR" ]; then
    file_count=$(find "$OUTPUT_DIR" -name "*.wav" | wc -l)
    echo "Total files created: $file_count"
fi
