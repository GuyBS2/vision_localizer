import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import csv
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from src.feature_extractor import FeatureExtractor

# ------------------------------
# Configuration
# ------------------------------

IMAGE_DIR = Path("data/map_images")
OUTPUT_FEATURES = Path("data/map_features.npy")
OUTPUT_FILENAMES = Path("data/map_filenames.csv")


# ------------------------------
# Feature Extraction Pipeline
# ------------------------------

def extract_features_from_directory(image_dir: Path, model: FeatureExtractor):
    """
    Extracts features from all .jpg images in the given directory.

    Args:
        image_dir: Directory containing input images.
        model: FeatureExtractor instance.

    Returns:
        Tuple of:
            - features: List of feature vectors (NumPy arrays)
            - filenames: List of corresponding filenames
    """
    image_paths = sorted(image_dir.glob("*.jpg"))
    print(f"Found {len(image_paths)} images to process.")

    features = []
    filenames = []

    for img_path in tqdm(image_paths, desc="Extracting features"):
        try:
            image = Image.open(img_path).convert("RGB")
            tensor = model.preprocess_image(image)
            vector = model(tensor)
            features.append(vector.numpy())
            filenames.append(img_path.name)
        except Exception as e:
            print(f"⚠️ Skipping {img_path.name}: {e}")

    return features, filenames


# ------------------------------
# Save Results
# ------------------------------

def save_features_to_disk(features: list, filenames: list, out_vec: Path, out_csv: Path):
    """
    Save extracted features and filenames to disk.

    Args:
        features: List of NumPy feature vectors.
        filenames: List of image filenames.
        out_vec: Path to .npy file for features.
        out_csv: Path to .csv file for filenames.
    """
    print(f"Saving {len(features)} feature vectors...")

    np.save(out_vec, np.stack(features).astype(np.float32))

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename"])
        for name in filenames:
            writer.writerow([name])

    print(f"✅ Done: saved {out_vec.name} and {out_csv.name}")


# ------------------------------
# Main
# ------------------------------

def main():
    model = FeatureExtractor()
    features, filenames = extract_features_from_directory(IMAGE_DIR, model)
    save_features_to_disk(features, filenames, OUTPUT_FEATURES, OUTPUT_FILENAMES)


if __name__ == "__main__":
    main()
