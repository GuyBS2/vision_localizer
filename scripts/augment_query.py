import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import random
import os
from PIL import Image, ImageEnhance, ImageFilter

# ------------------------------
# Augmentation Functions
# ------------------------------

def apply_blur(img):
    radius = random.uniform(1.0, 2.5)
    return img.filter(ImageFilter.GaussianBlur(radius))


def apply_rotate(img):
    angle = random.uniform(-15, 15)
    return img.rotate(angle, resample=Image.BILINEAR, expand=True)


def apply_crop(img):
    width, height = img.size
    crop_ratio = random.uniform(0.85, 0.95)
    new_w = int(width * crop_ratio)
    new_h = int(height * crop_ratio)
    x = random.randint(0, width - new_w)
    y = random.randint(0, height - new_h)
    cropped = img.crop((x, y, x + new_w, y + new_h))
    return cropped.resize((width, height))


def apply_shift(img):
    width, height = img.size
    max_dx = int(width * 0.1)
    max_dy = int(height * 0.1)
    dx = random.randint(-max_dx, max_dx)
    dy = random.randint(-max_dy, max_dy)
    return img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy))


def apply_brightness(img):
    enhancer = ImageEnhance.Brightness(img)
    factor = random.uniform(0.7, 1.3)
    return enhancer.enhance(factor)


# ------------------------------
# Augmentation Pipeline
# ------------------------------

AUGMENTATIONS = [
    apply_blur,
    apply_rotate,
    apply_crop,
    apply_shift,
    apply_brightness
]


def apply_random_augmentations(img):
    """
    Apply a random subset of augmentations to the image.
    Returns the augmented image and a list of operations applied.
    """
    augmented = img.copy()
    selected = random.sample(AUGMENTATIONS, k=random.randint(1, len(AUGMENTATIONS)))
    for func in selected:
        augmented = func(augmented)
    return augmented, [func.__name__ for func in selected]


# ------------------------------
# Main CLI
# ------------------------------

def main():
    parser = argparse.ArgumentParser(description="Apply random augmentations to a query image")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image")
    parser.add_argument("--num", type=int, default=5, help="Number of augmented images to create")
    parser.add_argument("--output_dir", type=str, default="data/test_inputs", help="Output directory")

    args = parser.parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    img = Image.open(input_path).convert("RGB")
    base_name = input_path.stem

    for i in range(args.num):
        aug_img, ops = apply_random_augmentations(img)
        out_name = f"{base_name}_aug{i}.jpg"
        out_path = output_dir / out_name
        aug_img.save(out_path)
        print(f"✔ Saved {out_name}  |  augmentations: {', '.join(ops)}")


if __name__ == "__main__":
    main()
