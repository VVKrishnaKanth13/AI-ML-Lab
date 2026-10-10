
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from config import IMAGE_DIR, RESULTS_DIR
from visualize import decode_segmentation_mask


def main():
    image_path = IMAGE_DIR / "bird.jpg"
    fcn_mask_path = RESULTS_DIR / "predicted_mask.pt"
    deeplab_mask_path = RESULTS_DIR / "deeplabv3_predicted_mask.pt"

    for path in [image_path, fcn_mask_path, deeplab_mask_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    original_image = np.array(
        Image.open(image_path).convert("RGB")
    )

    fcn_mask = torch.load(
        fcn_mask_path, map_location="cpu", weights_only=True
    ).numpy()

    deeplab_mask = torch.load(
        deeplab_mask_path, map_location="cpu", weights_only=True
    ).numpy()

    if fcn_mask.shape != deeplab_mask.shape:
        raise ValueError(
            f"Mask shapes differ: FCN={fcn_mask.shape}, "
            f"DeepLabV3={deeplab_mask.shape}"
        )

    fcn_colored = decode_segmentation_mask(fcn_mask)
    deeplab_colored = decode_segmentation_mask(deeplab_mask)

    # True where the two models predicted different class IDs
    difference = fcn_mask != deeplab_mask

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")

    axes[1].imshow(fcn_colored)
    axes[1].set_title("FCN ResNet101")

    axes[2].imshow(deeplab_colored)
    axes[2].set_title("DeepLabV3 ResNet101")

    axes[3].imshow(difference, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Different Predictions")

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()
    plt.show()

    total_pixels = difference.size
    different_pixels = int(difference.sum())
    agreement = 100 * (1 - different_pixels / total_pixels)

    print(f"FCN class IDs: {np.unique(fcn_mask).tolist()}")
    print(f"DeepLabV3 class IDs: {np.unique(deeplab_mask).tolist()}")
    print(f"Pixels with different predictions: {different_pixels}")
    print(f"Pixel-wise agreement: {agreement:.2f}%")


if __name__ == "__main__":
    main()