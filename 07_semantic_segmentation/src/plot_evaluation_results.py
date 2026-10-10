
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
from torchvision.models.segmentation import (
    FCN_ResNet101_Weights,
    DeepLabV3_ResNet101_Weights,
)

from config import DEVICE, IMAGE_SIZE, MEAN, STD, RESULTS_DIR


PROJECT_ROOT = RESULTS_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "voc_subset"
IMAGE_DIR = DATA_DIR / "images"
MASK_DIR = DATA_DIR / "masks"

NUM_CLASSES = 21
IGNORE_INDEX = 255

CLASS_NAMES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "dining table", "dog", "horse", "motorbike", "person",
    "potted plant", "sheep", "sofa", "train", "tv/monitor",
]

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

    return {
        "FCN ResNet101": fcn,
        "DeepLabV3 ResNet101": deeplab,
    }


def calculate_confusion_matrix(prediction, target):
    valid = (
            (target != IGNORE_INDEX)
            & (target >= 0)
            & (target < NUM_CLASSES)
    )

    prediction = prediction[valid]
    target = target[valid]

    indices = target * NUM_CLASSES + prediction

    return torch.bincount(
        indices,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def calculate_iou(confusion_matrix):
    matrix = confusion_matrix.astype(np.float64)

    true_positive = np.diag(matrix)
    ground_truth_count = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)

    union = ground_truth_count + predicted_count - true_positive

    scores = np.full(NUM_CLASSES, np.nan)
    valid_classes = union > 0
    scores[valid_classes] = (
            true_positive[valid_classes] / union[valid_classes]
    )

    return scores


def evaluate_model(model, image_paths):
    confusion_matrix = torch.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=torch.int64,
    )

    with torch.inference_mode():
        for image_path in image_paths:
            mask_path = MASK_DIR / f"{image_path.stem}.png"

            image = Image.open(image_path).convert("RGB")
            target_array = np.array(
                Image.open(mask_path),
                dtype=np.int64,
            )

            input_tensor = transform(image).unsqueeze(0).to(DEVICE)
            output = model(input_tensor)["out"]

            output = torch.nn.functional.interpolate(
                output,
                size=target_array.shape,
                mode="bilinear",
                align_corners=False,
            )

            prediction = output.argmax(dim=1).squeeze(0).cpu()
            target = torch.from_numpy(target_array)

            confusion_matrix += calculate_confusion_matrix(
                prediction, target
            )

    return calculate_iou(confusion_matrix.numpy())


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))

    if not image_paths:
        raise FileNotFoundError(f"No images found in {IMAGE_DIR}")

    for image_path in image_paths:
        if not (MASK_DIR / f"{image_path.stem}.png").exists():
            raise FileNotFoundError(
                f"Missing mask for {image_path.name}"
            )

    print(f"Evaluating {len(image_paths)} image-mask pairs")

    models_to_evaluate = load_models()
    results = {}

    for name, model in models_to_evaluate.items():
        print(f"Evaluating {name}...")
        results[name] = evaluate_model(model, image_paths)

        valid_scores = results[name][~np.isnan(results[name])]
        mean_iou = valid_scores.mean() if len(valid_scores) else float("nan")
        print(f"{name} mIoU: {mean_iou:.4f}")

    # Plot per-class IoU for both models.
    x = np.arange(NUM_CLASSES)
    width = 0.38

    fig, ax = plt.subplots(figsize=(16, 7))

    ax.bar(
        x - width / 2,
        np.nan_to_num(results["FCN ResNet101"], nan=0.0),
        width,
        label="FCN ResNet101",
        )

    ax.bar(
        x + width / 2,
        np.nan_to_num(results["DeepLabV3 ResNet101"], nan=0.0),
        width,
        label="DeepLabV3 ResNet101",
        )

    ax.set_xlabel("VOC class")
    ax.set_ylabel("Intersection over Union (IoU)")
    ax.set_title("Per-Class IoU: FCN vs DeepLabV3")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=55, ha="right")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()

    output_path = RESULTS_DIR / "per_class_iou_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Saved chart: {output_path}")


if __name__ == "__main__":
    main()
