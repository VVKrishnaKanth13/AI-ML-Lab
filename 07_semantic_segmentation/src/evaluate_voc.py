
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.models.segmentation import (
    FCN_ResNet101_Weights,
    DeepLabV3_ResNet101_Weights,
)

from config import DEVICE, MEAN, STD, RESULTS_DIR, IMAGE_SIZE


DATA_DIR = RESULTS_DIR.parent / "data" / "voc_subset"
IMAGE_DIR = DATA_DIR / "images"
MASK_DIR = DATA_DIR / "masks"

NUM_CLASSES = 21
IGNORE_INDEX = 255

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
    valid = target != IGNORE_INDEX
    prediction = prediction[valid]
    target = target[valid]

    indices = target * NUM_CLASSES + prediction

    return torch.bincount(
        indices,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def calculate_metrics(confusion_matrix):
    confusion_matrix = confusion_matrix.astype(np.float64)

    true_positive = np.diag(confusion_matrix)
    ground_truth_count = confusion_matrix.sum(axis=1)
    predicted_count = confusion_matrix.sum(axis=0)

    union = ground_truth_count + predicted_count - true_positive

    iou = np.full(NUM_CLASSES, np.nan)
    valid_classes = union > 0
    iou[valid_classes] = (
            true_positive[valid_classes] / union[valid_classes]
    )

    mean_iou = np.nanmean(iou) if valid_classes.any() else float("nan")

    pixel_accuracy = (
        true_positive.sum() / confusion_matrix.sum()
        if confusion_matrix.sum() > 0
        else float("nan")
    )

    return iou, mean_iou, pixel_accuracy


def evaluate_model(model, image_paths):
    confusion_matrix = torch.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=torch.int64,
    )

    with torch.inference_mode():
        for image_path in image_paths:
            mask_path = MASK_DIR / f"{image_path.stem}.png"

            image = Image.open(image_path).convert("RGB")
            ground_truth = np.array(
                Image.open(mask_path),
                dtype=np.int64,
            )

            input_tensor = transform(image).unsqueeze(0).to(DEVICE)

            output = model(input_tensor)["out"]

            # Resize predictions to the original ground-truth dimensions.
            output = torch.nn.functional.interpolate(
                output,
                size=ground_truth.shape,
                mode="bilinear",
                align_corners=False,
            )

            prediction = output.argmax(dim=1).squeeze(0).cpu()

            target = torch.from_numpy(ground_truth)

            valid = target != IGNORE_INDEX
            target_valid = target[valid]
            prediction_valid = prediction[valid]

            # Exclude invalid target IDs before building the confusion matrix.
            valid_ids = (
                    (target_valid >= 0)
                    & (target_valid < NUM_CLASSES)
                    & (prediction_valid >= 0)
                    & (prediction_valid < NUM_CLASSES)
            )

            indices = (
                    target_valid[valid_ids] * NUM_CLASSES
                    + prediction_valid[valid_ids]
            )

            confusion_matrix += torch.bincount(
                indices,
                minlength=NUM_CLASSES * NUM_CLASSES,
            ).reshape(NUM_CLASSES, NUM_CLASSES)

    return confusion_matrix.numpy()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))

    if not image_paths:
        raise FileNotFoundError(f"No images found in {IMAGE_DIR}")

    for image_path in image_paths:
        if not (MASK_DIR / f"{image_path.stem}.png").exists():
            raise FileNotFoundError(
                f"Matching ground-truth mask not found for {image_path.name}"
            )

    print(f"Evaluating {len(image_paths)} image-mask pairs on {DEVICE}")

    models_to_evaluate = load_models()

    for model_name, model in models_to_evaluate.items():
        print(f"\nEvaluating {model_name}...")

        confusion_matrix = evaluate_model(model, image_paths)
        iou, mean_iou, pixel_accuracy = calculate_metrics(confusion_matrix)

        print(f"Mean IoU      : {mean_iou:.4f}")
        print(f"Pixel accuracy: {pixel_accuracy:.4f}")

        for class_id, score in enumerate(iou):
            if np.isnan(score):
                print(f"Class {class_id:2d}: not present in evaluated union")
            else:
                print(f"Class {class_id:2d}: IoU = {score:.4f}")


if __name__ == "__main__":
    main()
