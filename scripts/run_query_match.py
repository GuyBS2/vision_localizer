import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
from PIL import Image
import matplotlib.pyplot as plt

from src.feature_extractor import FeatureExtractor
from src.matcher import QueryMatcher


def load_query_image(query_path: Path) -> Image.Image:
    """
    Loads and validates the query image.
    """
    if not query_path.exists():
        raise FileNotFoundError(f"Query image not found: {query_path}")
    return Image.open(query_path).convert("RGB")


def visualize_result(query_img: Image.Image, matched_img: Image.Image, match_filename: str):
    """
    Displays the query image and its top match side by side.
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(query_img)
    plt.title("Query Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(matched_img)
    plt.title(f"Top Match\n{match_filename}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Find best match for a query image")
    parser.add_argument("--query", type=str, required=True, help="Path to the query image")
    args = parser.parse_args()

    # Load and process query image
    query_path = Path(args.query)
    query_img = load_query_image(query_path)

    extractor = FeatureExtractor()
    query_tensor = extractor.preprocess_image(query_img)
    query_feature = extractor(query_tensor)  # shape: [1280]

    # Load matcher and run comparison
    matcher = QueryMatcher(
        feature_file="data/map_features.npy",
        filename_file="data/map_filenames.csv",
        metric="cosine"
    )

    match_filename, similarity = matcher.match(query_feature, top_k=1)[0]
    print(f"\n📍 Best match: {match_filename} (score: {similarity:.4f})")

    matched_img = Image.open(Path("data/map_images") / match_filename)
    visualize_result(query_img, matched_img, match_filename)


if __name__ == "__main__":
    main()
