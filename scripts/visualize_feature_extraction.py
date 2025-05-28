import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.feature_extractor import FeatureExtractor

# ------------------------------
# Configuration
# ------------------------------

IMAGE_DIR = Path("data/map_images")
SAMPLE_IMAGE = sorted(IMAGE_DIR.glob("*.jpg"))[0]  # Pick the first available tile


# ------------------------------
# Visualization Utilities
# ------------------------------

def unnormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Undo ImageNet normalization and convert tensor to NumPy for display.

    Args:
        tensor: Torch tensor of shape [3, 224, 224]

    Returns:
        Image as NumPy array in shape [224, 224, 3]
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = tensor.permute(1, 2, 0).cpu().numpy()
    image = std * image + mean
    return np.clip(image, 0, 1)


def plot_feature_extraction(original_img: Image.Image, resized_img: np.ndarray, feature_vec: np.ndarray):
    """
    Display original tile, resized input, and extracted feature vector.

    Args:
        original_img: Raw input tile
        resized_img: Preprocessed image (after resizing and normalization)
        feature_vec: Extracted feature vector (1D)
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title("Original Tile (150×150)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(resized_img)
    plt.title("Resized & Normalized (224×224)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.plot(feature_vec)
    plt.title("Feature Vector (1280-Dim)")
    plt.xlabel("Dimension")
    plt.tight_layout()
    plt.show()


# ------------------------------
# Main
# ------------------------------

def main():
    # Load image
    img = Image.open(SAMPLE_IMAGE).convert("RGB")

    # Initialize model
    model = FeatureExtractor()

    # Preprocess
    tensor = model.preprocess_image(img)  # [1, 3, 224, 224]

    # Extract features
    feature = model(tensor).numpy()  # [1280]
    resized = unnormalize_tensor(tensor.squeeze())

    # Plot
    plot_feature_extraction(img, resized, feature)

    # Log
    print(f"Sample image: {SAMPLE_IMAGE.name}")
    print(f"Feature shape: {feature.shape}")
    print(f"Feature vector (first 10 dims): {np.round(feature[:10], 3)}")


if __name__ == "__main__":
    main()
