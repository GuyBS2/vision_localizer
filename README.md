# Vision Localizer

An AI-assisted visual localization system that estimates position by matching an input image to a pre-built database of map tiles. Designed for GPS-denied environments and scalable to real-world geolocation.

## Project Overview

This project simulates a visual navigation pipeline:
- Aerial or satellite images are divided into fixed-size map tiles
- Each tile is converted into a feature vector using a pretrained CNN
- A query image is matched to the most similar tile using feature similarity (cosine or L2)
- The system returns the best-matching tile and its known location

The current version uses synthetic (x, y) positions. Future extensions will include true GPS-aligned data.

## Project Structure

```
vision_localizer/
├── src/                         # Core logic: feature extractor, localizer
│   └── feature_extractor.py
├── scripts/                     # Dataset processing and utilities
│   ├── download_dataset.sh
│   ├── split_images_to_grid.py
│   └── extract_features_batch.py (coming soon)
├── data/                        # Raw dataset + image patches (ignored in Git)
│   ├── raw_mass_data/
│   └── map_images/
├── requirements.txt             # Python dependencies
├── setup_workspace.sh           # Complete workspace setup (one command setup)
├── README.md
└── .gitignore
```

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/GuyBS2/vision_localizer.git
cd vision_localizer
```

### 2. Setup the full environment (Python + Dataset + Patching)

```bash
./setup_workspace.sh
```

> This script:
> - Creates a Python virtual environment
> - Installs dependencies
> - Downloads the Massachusetts Buildings dataset from Kaggle
> - Generates 150×150 image patches from the raw satellite images

## Output Files (after setup)

- `data/map_images/`        – all 150x150 tiles
- `data/map_features.npy`   – feature vectors
- `data/map_filenames.csv`  – file name

## Dependencies

Tested on Python 3.12+. Install using:

This project is licensed under the MIT License.

## Credits

- Dataset: Massachusetts Buildings Dataset  
- Inspired by: [Mnih, Volodymyr (2013) – Machine Learning for Aerial Image Labeling](https://www.cs.toronto.edu/~vmnih/docs/MnihThesis.pdf)