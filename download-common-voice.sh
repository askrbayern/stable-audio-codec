#!/usr/bin/bash

set -e


URL="https://storage.googleapis.com/common-voice-prod-prod-datasets/cv-corpus-22.0-2025-06-20/cv-corpus-22.0-2025-06-20-en.tar.gz?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gke-prod%40moz-fx-common-voice-prod.iam.gserviceaccount.com%2F20250824%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250824T111627Z&X-Goog-Expires=43200&X-Goog-SignedHeaders=host&X-Goog-Signature=6038be49f92719ff8eeb76d8749e1f6ad9d4aeeb51c589de846aba59fe72a62155069fdc8fea8aff9324b5fb80db83650a86e4b2b33e026a331e5f3b3d38c783629327319bff11068e7817d2922a3d688cd0c19a3562b3bd92f6567b494b470250ce740b0146ecf2f9002eebb90edb798a247af2b979fcab89d58eb91b9ca2d873f54196d52f32016f5ba010e7189b36a0c6d9ee4ec3bf3f69ab00378fc2d62a7a730980e22b81d39ce11c7fbc160386d6ba0a4b0efdf17f52fa2536b1463d27240c747172c1f7995d22ac95fdc2b49880a71f3aa606e0af57e94bf12db8750805ff7b8ff2dc342b544f8e69d2cedb3c7eaaa1fd6934d25c5e26326df5a781aa"

OUTPUT_PATH="/itet-stor/feigao/net_scratch/datasets/common_voice"

# Create the save directory if it doesn't exist
mkdir -p "$OUTPUT_PATH"

# Get the filename from the URL
FILENAME=$(basename "${URL%%\?*}")

echo "Starting to download Common Voice dataset..."
echo "Saving to: $OUTPUT_PATH/$FILENAME"

# Download the file directly into the specified directory
wget -c -O "$OUTPUT_PATH/$FILENAME" "$URL"

echo "Download completed!"
echo "----------------------------------------------------"
echo "Starting to unzip file in: $OUTPUT_PATH"

# Unzip the file. The -C flag tells tar to change to that directory before unzipping.
tar -xzvf "$OUTPUT_PATH/$FILENAME" -C "$OUTPUT_PATH"

echo "Unzip completed! Your data is in $OUTPUT_PATH"

