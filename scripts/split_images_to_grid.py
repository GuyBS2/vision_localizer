from pathlib import Path
from PIL import Image, UnidentifiedImageError

# ------------------------------
# Configuration
# ------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data/raw_mass_data/png/train"
OUTPUT_DIR = PROJECT_ROOT / "data/map_images"
PATCH_SIZE = 150

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------
# Patch Extraction
# ------------------------------

def split_image_to_patches(image_path: Path, patch_size: int) -> int:
    """
    Splits an image into fixed-size patches and saves them to disk.

    Args:
        image_path: Path to input image
        patch_size: Size of square patch (e.g., 150)

    Returns:
        Number of patches saved
    """
    try:
        base_name = image_path.stem
        img = Image.open(image_path).convert("RGB")
        width, height = img.size

        count = 0
        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                box = (x, y, x + patch_size, y + patch_size)
                patch = img.crop(box)
                if patch.size == (patch_size, patch_size):
                    filename = f"{base_name}_{x}_{y}.jpg"
                    patch.save(OUTPUT_DIR / filename)
                    count += 1

        print(f"✔ {image_path.name} → {count} patches")
        return count

    except UnidentifiedImageError:
        print(f"⚠ Skipped unreadable image: {image_path.name}")
        return 0


# ------------------------------
# Main
# ------------------------------

if __name__ == "__main__":
    image_paths = sorted(INPUT_DIR.glob("*.png"))
    print(f"Found {len(image_paths)} images to process.\n")

    total_patches = 0
    for img_path in image_paths:
        total_patches += split_image_to_patches(img_path, PATCH_SIZE)

    print(f"\n Done. Total patches saved: {total_patches}")
