import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import csv
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from src.feature_extractor import FeatureExtractor

# Configuration
IMAGE_DIR = Path("data/map_images")
OUTPUT_FEATURES = Path("data/map_features.npy")
OUTPUT_FILENAMES = Path("data/map_filenames.csv")

# Initialize model
model = FeatureExtractor()

# Collect all images
image_paths = sorted(IMAGE_DIR.glob("*.jpg"))
print(f"Found {len(image_paths)} images to process.")

# Feature extraction
all_features = []
all_filenames = []

for img_path in tqdm(image_paths, desc="Extracting features"):
    try:
        image = Image.open(img_path).convert("RGB")
        tensor = model.preprocess_image(image)
        feature = model(tensor)
        all_features.append(feature.numpy())
        all_filenames.append(img_path.name)
    except Exception as e:
        print(f"Skipping {img_path.name}: {e}")

# Save outputs
print(f"Saving {len(all_features)} feature vectors...")

np.save(OUTPUT_FEATURES, np.stack(all_features).astype(np.float32))

with open(OUTPUT_FILENAMES, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename"])
    for name in all_filenames:
        writer.writerow([name])

print("Done: saved map_features.npy and map_filenames.csv.")
