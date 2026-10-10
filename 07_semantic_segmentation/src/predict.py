
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.models.segmentation import FCN_ResNet101_Weights

from config import (
    DEVICE,
    IMAGE_SIZE,
    MEAN,
    STD,
    NUM_CLASSES,
    IMAGE_DIR,
    RESULTS_DIR,
)


def load_model():
    weights = FCN_ResNet101_Weights.DEFAULT

    model = models.segmentation.fcn_resnet101(
        weights=weights
    )

    model = model.to(DEVICE)
    model.eval()

    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MEAN,
            std=STD,
        ),
    ])


def predict(model, image_path):
    image = Image.open(image_path).convert("RGB")

    transform = get_transform()
    input_tensor = transform(image)

    # Add the batch dimension: [C, H, W] -> [1, C, H, W]
    input_batch = input_tensor.unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        output = model(input_batch)["out"]

    # Output: [batch, classes, height, width]
    print(f"Input shape  : {input_batch.shape}")
    print(f"Output shape : {output.shape}")

    # Select the highest-scoring class for every pixel.
    predicted_mask = torch.argmax(output, dim=1)

    # Remove the batch dimension.
    predicted_mask = predicted_mask.squeeze(0).cpu()

    print(f"Mask shape   : {predicted_mask.shape}")
    print(f"Class IDs    : {torch.unique(predicted_mask).tolist()}")

    return image, predicted_mask


def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    image_path = IMAGE_DIR / "bird.jpg"

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        print("Place an image named bird.jpg inside data/images/")
        return

    model = load_model()
    image, predicted_mask = predict(model, image_path)

    # Save the class-ID mask for the next visualization step.
    mask_path = RESULTS_DIR / "predicted_mask.pt"
    torch.save(predicted_mask, mask_path)

    print(f"Saved class-ID mask: {mask_path}")


if __name__ == "__main__":
    main()
