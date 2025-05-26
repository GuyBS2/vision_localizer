import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from src.feature_extractor import FeatureExtractor
from src.matcher import QueryMatcher

# === Parse arguments ===
parser = argparse.ArgumentParser(description="Find best match for a query image")
parser.add_argument("--query", type=str, required=True, help="Path to the query image")
args = parser.parse_args()

# === Load query image ===
query_path = Path(args.query)
if not query_path.exists():
    raise FileNotFoundError(f"Query image not found: {query_path}")

query_img = Image.open(query_path).convert("RGB")

# === Load feature extractor and matcher ===
extractor = FeatureExtractor()
query_tensor = extractor.preprocess_image(query_img)
query_feature = extractor(query_tensor)  # shape: [1280]

matcher = QueryMatcher(
    feature_file="data/map_features.npy",
    filename_file="data/map_filenames.csv",
    metric="cosine"
)

# === Match
top_match = matcher.match(query_feature, top_k=1)[0]
match_filename, similarity = top_match

print(f"\n📍 Best match: {match_filename} (score: {similarity:.4f})")

# === Show query and result
matched_img = Image.open(Path("data/map_images") / match_filename)

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