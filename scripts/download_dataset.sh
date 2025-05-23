#!/bin/bash

set -e

# Constants
DATA_DIR="data"
RAW_DIR="$DATA_DIR/raw_mass_data"
ZIP_FILE="$DATA_DIR/massachusetts-buildings-dataset.zip"

echo "Creating data directory if not exists..."
mkdir -p "$DATA_DIR"

echo "Downloading Massachusetts Buildings Dataset from Kaggle..."
kaggle datasets download -d balraj98/massachusetts-buildings-dataset -p "$DATA_DIR"

echo "Unzipping dataset..."
unzip "$ZIP_FILE" -d "$RAW_DIR"

echo "Removing ZIP file..."
rm "$ZIP_FILE"

echo "Dataset downloaded and extracted to: $RAW_DIR"