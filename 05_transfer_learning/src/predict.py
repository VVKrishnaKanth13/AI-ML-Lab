import sys

import torch
from PIL import Image

from config import DEVICE
from dataset import valid_test_transform
from model import TransferLearningModel


def main():

    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        return

    image_path = sys.argv[1]

    device = torch.device(
        DEVICE if torch.cuda.is_available() else "cpu"
    )

    # Load class names
    from dataset import train_data

    class_names = train_data.classes

    # Create model
    model = TransferLearningModel(
        num_classes=len(class_names)
    )

    # Load trained weights
    model.load_state_dict(
        torch.load(
            "../models/best_model.pth",
            map_location=device,
            weights_only=True
        )
    )

    model = model.to(device)
    model.eval()

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Apply validation/test preprocessing
    image = valid_test_transform(image)

    # Add batch dimension
    image = image.unsqueeze(0)

    image = image.to(device)

    # Inference
    with torch.inference_mode():

        outputs = model(image)

        probabilities = torch.exp(outputs)

        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=5,
            dim=1
        )

    print("\nPrediction")
    print("-" * 30)

    for rank in range(5):

        class_index = top_indices[0][rank].item()
        probability = top_probabilities[0][rank].item()

        print(
            f"{rank + 1}. "
            f"{class_names[class_index]}: "
            f"{probability * 100:.2f}%"
        )


if __name__ == "__main__":
    main()