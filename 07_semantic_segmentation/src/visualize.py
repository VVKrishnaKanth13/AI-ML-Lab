import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from config import IMAGE_DIR, RESULTS_DIR

LABEL_COLORS = np.array([
    (0, 0, 0),          # 0: background
    (128, 0, 0),        # 1: aeroplane
    (0, 128, 0),        # 2: bicycle
    (128, 128, 0),      # 3: bird
    (0, 0, 128),        # 4: boat
    (128, 0, 128),      # 5: bottle
    (0, 128, 128),      # 6: bus
    (128, 128, 128),    # 7: car
    (64, 0, 0),         # 8: cat
    (192, 0, 0),        # 9: chair
    (64, 128, 0),       # 10: cow
    (192, 128, 0),      # 11: dining table
    (64, 0, 128),       # 12: dog
    (192, 0, 128),      # 13: horse
    (64, 128, 128),     # 14: motorbike
    (192, 128, 128),    # 15: person
    (0, 64, 0),         # 16: potted plant
    (128, 64, 0),       # 17: sheep
    (0, 192, 0),        # 18: sofa
    (128, 192, 0),      # 19: train
    (0, 64, 128),       # 20: tv/monitor
], dtype=np.uint8)


def decode_segmentation_mask(mask):
    mask = np.asarray(mask, dtype=np.uint8)

    # Initialize all pixels as black.
    colored_mask = np.zeros(
        (*mask.shape, 3),
        dtype=np.uint8,
    )

    # Convert only valid class IDs (0-20) to RGB colors.
    valid_pixels = mask < len(LABEL_COLORS)
    colored_mask[valid_pixels] = LABEL_COLORS[mask[valid_pixels]]

    # Pixels with ID 255 remain black.
    return colored_mask


def main():
    image_path = IMAGE_DIR / "bird.jpg"
    mask_path = RESULTS_DIR / "predicted_mask.pt"

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    # Load original image
    original_image = Image.open(image_path).convert("RGB")
    original_array = np.array(original_image)

    # Load predicted class-ID mask
    mask = torch.load(
        mask_path,
        map_location="cpu",
        weights_only=True
    ).numpy()

    # Convert class IDs into RGB colors
    colored_mask = decode_segmentation_mask(mask)

    # Resize mask if its dimensions differ from the original image
    if colored_mask.shape[:2] != original_array.shape[:2]:
        colored_mask = np.array(
            Image.fromarray(colored_mask).resize(
                (original_array.shape[1], original_array.shape[0]),
                Image.Resampling.NEAREST
            )
        )

    # Blend original image and colored mask
    alpha = 0.45
    overlay = (
            (1 - alpha) * original_array.astype(np.float32)
            + alpha * colored_mask.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    # Save colored mask and overlay
    colored_mask_path = RESULTS_DIR / "segmentation_result.png"
    overlay_path = RESULTS_DIR / "segmentation_overlay.png"

    Image.fromarray(colored_mask).save(colored_mask_path)
    Image.fromarray(overlay).save(overlay_path)

    # Display all three images
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    axes[0].imshow(original_array)
    axes[0].set_title("Original Image")

    axes[1].imshow(colored_mask)
    axes[1].set_title("Predicted Segmentation Mask")

    axes[2].imshow(overlay)
    axes[2].set_title("Segmentation Overlay")

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()
    plt.show()

    print(f"Original image shape : {original_array.shape}")
    print(f"Mask shape           : {mask.shape}")
    print(f"Colored mask shape   : {colored_mask.shape}")
    print(f"Saved colored mask   : {colored_mask_path}")
    print(f"Saved overlay        : {overlay_path}")


if __name__ == "__main__":
    main()