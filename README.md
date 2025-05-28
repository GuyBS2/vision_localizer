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
├── src/                         # Core logic: feature extractor, localizer, matcher
│   ├── feature_extractor.py
│   └── matcher.py
├── scripts/                     # Dataset processing and utilities
│   ├── download_dataset.sh
│   ├── extract_features_batch.py
│   ├── query_image_match.py
│   ├── split_images_to_grid.py
│   ├── augment_query.py
│   └── visualize_feature_extraction.py
├── data/                        # Raw dataset + image patches (ignored in Git)
│   ├── raw_mass_data/
│   ├── map_images/
│   ├── map_features.npy
│   ├── map_filenames.csv
│   └── test_inputs/
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
> - Extracts feature vectors for all patches

## Output Files (after setup)

- `data/map_images/`        – all 150x150 tiles
- `data/map_features.npy`   – feature vectors
- `data/map_filenames.csv`  – filenames associated with each vector
- `data/test_inputs/`       – query images to match against the map

## Dependencies

Tested on Python 3.12+.  
Install Python dependencies using:


## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Credits

- Dataset: Massachusetts Buildings Dataset  
- Inspired by: [Mnih, Volodymyr (2013) – Machine Learning for Aerial Image Labeling](https://www.cs.toronto.edu/~vmnih/docs/MnihThesis.pdf)