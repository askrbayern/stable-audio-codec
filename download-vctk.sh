#!/usr/bin/bash

set -e


URL="https://datashare.ed.ac.uk/download/DS_10283_3443.zip"

OUTPUT_PATH="/itet-stor/feigao/net_scratch/datasets/vctk"

# Create the save directory if it doesn't exist
mkdir -p "$OUTPUT_PATH"

# Get the filename from the URL
FILENAME=$(basename "${URL%%\?*}")

echo "Starting to download VCTK dataset..."
echo "Saving to: $OUTPUT_PATH/$FILENAME"

# Download the file directly into the specified directory
wget -c -O "$OUTPUT_PATH/$FILENAME" "$URL"

echo "Download completed!"
echo "----------------------------------------------------"
echo "Starting to unzip file in: $OUTPUT_PATH"

# Unzip the file. The -C flag tells tar to change to that directory before unzipping.
tar -xzvf "$OUTPUT_PATH/$FILENAME" -C "$OUTPUT_PATH"

echo "Unzip completed! Your data is in $OUTPUT_PATH"

