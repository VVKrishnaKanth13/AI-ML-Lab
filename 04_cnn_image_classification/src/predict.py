import sys

import torch
from PIL import Image

from dataset import val_data, CLASS_MAPPING
from model import MonkeyCNN


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "../models/best.pt"


def predict(image_path):

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Same preprocessing used during validation
    image = val_data.transform(image)

    # Add batch dimension
    image = image.unsqueeze(0)

    # Move image to device
    image = image.to(DEVICE)

    # Create model
    model = MonkeyCNN(num_classes=10).to(DEVICE)

    # Load trained weights
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.eval()

    # Inference
    with torch.inference_mode():

        outputs = model(image)

        # Convert logits → probabilities
        probabilities = torch.softmax(outputs, dim=1)

        # Get top 5 predictions
        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=5,
            dim=1
        )

    print("\nTop 5 Predictions:")

    for probability, index in zip(
            top_probabilities[0],
            top_indices[0]
    ):

        class_index = index.item()
        class_name = CLASS_MAPPING[class_index]
        confidence = probability.item() * 100

        print(
            f"{class_name}: {confidence:.2f}%"
        )


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    predict(sys.argv[1])