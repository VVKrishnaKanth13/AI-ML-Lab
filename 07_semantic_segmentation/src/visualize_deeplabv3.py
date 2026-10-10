import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from config import IMAGE_DIR, RESULTS_DIR

# Reuse the same VOC color palette as visualize.py
from visualize import decode_segmentation_mask


def main():
    image_path = IMAGE_DIR / "bird.jpg"
    mask_path = RESULTS_DIR / "deeplabv3_predicted_mask.pt"

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    original_image = Image.open(image_path).convert("RGB")
    original_array = np.array(original_image)

    mask = torch.load(
        mask_path,
        map_location="cpu",
        weights_only=True
    ).numpy()

    colored_mask = decode_segmentation_mask(mask)

    # Match the original image dimensions if necessary
    if colored_mask.shape[:2] != original_array.shape[:2]:
        colored_mask = np.array(
            Image.fromarray(colored_mask).resize(
                (original_array.shape[1], original_array.shape[0]),
                Image.Resampling.NEAREST
            )
        )

    # Blend the image with the predicted mask
    alpha = 0.45
    overlay = (
            (1 - alpha) * original_array.astype(np.float32)
            + alpha * colored_mask.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    overlay_path = RESULTS_DIR / "deeplabv3_overlay.png"
    Image.fromarray(overlay).save(overlay_path)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    axes[0].imshow(original_array)
    axes[0].set_title("Original Image")

    axes[1].imshow(colored_mask)
    axes[1].set_title("DeepLabV3 Mask")

    axes[2].imshow(overlay)
    axes[2].set_title("DeepLabV3 Overlay")

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()
    plt.show()

    print(f"Mask shape: {mask.shape}")
    print(f"Class IDs: {np.unique(mask).tolist()}")
    print(f"Saved overlay: {overlay_path}")


if __name__ == "__main__":
    main()
