from pathlib import Path
from PIL import Image, UnidentifiedImageError
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data/raw_mass_data/png/train"
OUTPUT_DIR = PROJECT_ROOT / "data/map_images"
PATCH_SIZE = 150

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image_paths = sorted(INPUT_DIR.glob("*.png"))
print(f"Found {len(image_paths)} images to process.")

total_patches = 0

for image_path in image_paths:
    try:
        base_name = image_path.stem
        img = Image.open(image_path).convert("RGB")
        width, height = img.size

        for y in range(0, height, PATCH_SIZE):
            for x in range(0, width, PATCH_SIZE):
                box = (x, y, x + PATCH_SIZE, y + PATCH_SIZE)
                patch = img.crop(box)
                if patch.size == (PATCH_SIZE, PATCH_SIZE):
                    patch_filename = f"{base_name}_{x}_{y}.jpg"
                    patch.save(OUTPUT_DIR / patch_filename)
                    total_patches += 1

        print(f"Processed {image_path.name}, patches created.")

    except UnidentifiedImageError:
        print(f"Skipped unreadable image: {image_path.name}")

print(f"Done. Total patches saved: {total_patches}")