#!/bin/bash

set -e

echo "Creating virtual environment (.venv)..."
python3 -m venv .venv
source .venv/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Verifying Kaggle API credentials..."
if [ ! -f ~/.kaggle/kaggle.json ]; then
  echo "ERROR: Kaggle API key not found in ~/.kaggle/kaggle.json"
  echo "Please follow https://www.kaggle.com/docs/api to configure it."
  exit 1
fi

echo "Checking dataset status..."
if [ -d data/raw_mass_data ] && [ -f data/raw_mass_data/metadata.csv ]; then
  echo "Dataset already exists. Skipping download."
else
  echo "Downloading dataset from Kaggle..."
  bash scripts/download_dataset.sh
fi

echo "Checking for image patches..."
if [ -d data/map_images ] && [ "$(ls -A data/map_images)" ]; then
  echo "Image patches already exist. Skipping patch generation."
else
  echo "Generating image patches..."
  python3 scripts/split_images_to_grid.py
fi

echo ""
echo "Workspace is ready!"
echo "You can now continue with feature extraction or development."