
from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "voc_subset"

IMAGE_DIR = DATA_DIR / "images"
MASK_DIR = DATA_DIR / "masks"

NUM_IMAGES = 20


def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Pascal VOC 2012 validation dataset...")

    dataset = load_dataset(
        "shijli/voc2012",
        "segmentation",
        split="validation",
        streaming=True,
    )

    downloaded = 0

    for sample in dataset:
        image_id = sample["id"]

        image_path = IMAGE_DIR / f"{image_id}.jpg"
        mask_path = MASK_DIR / f"{image_id}.png"

        # Skip pairs already downloaded
        if image_path.exists() and mask_path.exists():
            downloaded += 1
            print(f"Already exists: {image_id}")
        else:
            sample["image"].convert("RGB").save(image_path)

            # Preserve the mask's class-ID pixel values.
            # Do not convert the mask to RGB.
            sample["mask"].save(mask_path)

            downloaded += 1
            print(f"Saved {downloaded}/{NUM_IMAGES}: {image_id}")

        if downloaded >= NUM_IMAGES:
            break

    print("\nDataset subset ready!")
    print(f"Images directory: {IMAGE_DIR}")
    print(f"Masks directory : {MASK_DIR}")
    print(f"Image-mask pairs: {downloaded}")


if __name__ == "__main__":
    main()
