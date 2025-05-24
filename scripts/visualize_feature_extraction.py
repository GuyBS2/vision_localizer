import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from src.feature_extractor import FeatureExtractor

# === Config ===

# path to a patch image folder
IMAGE_PATH = Path("data/map_images")
# pick first one
sample_image = sorted(IMAGE_PATH.glob("*.jpg"))[0]

# === Load image ===
img = Image.open(sample_image).convert("RGB")

# === Initialize model ===
model = FeatureExtractor()

# === Preprocess image ===
tensor = model.preprocess_image(img)  # [1, 3, 224, 224]

# === Extract feature vector ===
feature = model(tensor).numpy()  # [1280]

# === Plot original image ===
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Original Tile (150x150)")
plt.axis("off")

# === Plot resized image ===
resized_tensor = tensor.squeeze().permute(1, 2, 0).cpu().numpy()  # [224, 224, 3]

# Undo normalization for display
mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])
resized_img = std * resized_tensor + mean
resized_img = np.clip(resized_img, 0, 1)

plt.subplot(1, 3, 2)
plt.imshow(resized_img)
plt.title("Resized & Normalized (224x224)")
plt.axis("off")

# === Plot feature vector ===
plt.subplot(1, 3, 3)
plt.plot(feature)
plt.title("Feature Vector (1280-dim)")
plt.xlabel("Dimension")
plt.tight_layout()
plt.show()

# === Print summary ===
print("Sample image:", sample_image.name)
print("Feature shape:", feature.shape)
print("Feature vector (first 10 dims):", feature[:10])
