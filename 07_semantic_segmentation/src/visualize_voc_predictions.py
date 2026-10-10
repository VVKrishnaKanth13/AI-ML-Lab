
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
from torchvision.models.segmentation import (
    FCN_ResNet101_Weights,
    DeepLabV3_ResNet101_Weights,
)

from config import DEVICE, IMAGE_SIZE, MEAN, STD
from visualize import decode_segmentation_mask


PROJECT_ROOT = IMAGE_DIR = (
    __import__("pathlib").Path(__file__).resolve().parent.parent
)
DATA_DIR = PROJECT_ROOT / "data" / "voc_subset"
VOC_IMAGE_DIR = DATA_DIR / "images"
VOC_MASK_DIR = DATA_DIR / "masks"

NUM_IMAGES_TO_SHOW = 5

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def load_models():
    fcn = models.segmentation.fcn_resnet101(
        weights=FCN_ResNet101_Weights.DEFAULT
    ).to(DEVICE).eval()

    deeplab = models.segmentation.deeplabv3_resnet101(
        weights=DeepLabV3_ResNet101_Weights.DEFAULT
    ).to(DEVICE).eval()

    return fcn, deeplab


def predict_mask(model, image):
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        output = model(input_tensor)["out"]

        output = torch.nn.functional.interpolate(
            output,
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )

    return output.argmax(dim=1).squeeze(0).cpu().numpy()


def main():
    image_paths = sorted(VOC_IMAGE_DIR.glob("*.jpg"))

    if not image_paths:
        raise FileNotFoundError(
            f"No VOC images found in {VOC_IMAGE_DIR}"
        )

    fcn, deeplab = load_models()

    for image_path in image_paths[:NUM_IMAGES_TO_SHOW]:
        mask_path = VOC_MASK_DIR / f"{image_path.stem}.png"

        if not mask_path.exists():
            print(f"Skipping {image_path.name}: ground-truth mask missing")
            continue

        image = Image.open(image_path).convert("RGB")
        original = np.array(image)

        # VOC ground-truth mask contains class IDs, not RGB colors.
        ground_truth = np.array(
            Image.open(mask_path), dtype=np.uint8
        )

        fcn_mask = predict_mask(fcn, image)
        deeplab_mask = predict_mask(deeplab, image)

        ground_truth_rgb = decode_segmentation_mask(ground_truth)
        fcn_rgb = decode_segmentation_mask(fcn_mask)
        deeplab_rgb = decode_segmentation_mask(deeplab_mask)

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))

        axes[0].imshow(original)
        axes[0].set_title("Original Image")

        axes[1].imshow(ground_truth_rgb)
        axes[1].set_title("Ground Truth")

        axes[2].imshow(fcn_rgb)
        axes[2].set_title("FCN ResNet101")

        axes[3].imshow(deeplab_rgb)
        axes[3].set_title("DeepLabV3 ResNet101")

        for axis in axes:
            axis.axis("off")

        fig.suptitle(image_path.stem)
        plt.tight_layout()
        plt.show()

        print(f"\nImage: {image_path.name}")
        print(f"Ground-truth classes: {np.unique(ground_truth).tolist()}")
        print(f"FCN classes         : {np.unique(fcn_mask).tolist()}")
        print(f"DeepLabV3 classes   : {np.unique(deeplab_mask).tolist()}")


if __name__ == "__main__":
    main()
