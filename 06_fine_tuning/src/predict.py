import sys

import torch
from PIL import Image

from config import DEVICE
from dataset import valid_transform, CLASS_MAPPING, train_data
from model import FineTuningModel


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        return

    image_path = sys.argv[1]

    device = torch.device(
        DEVICE if torch.cuda.is_available() else "cpu"
    )

    # Use the same class order as the training dataset.
    class_names = train_data.classes

    model = FineTuningModel(
        num_classes=len(class_names)
    )

    model.load_state_dict(
        torch.load(
            "../models/best_model.pth",
            map_location=device,
            weights_only=True
        )
    )

    model = model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image = valid_transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(image)

        # Convert raw logits into probabilities.
        probabilities = torch.softmax(outputs, dim=1)

        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=min(5, len(class_names)),
            dim=1
        )

    print("\nPrediction")
    print("-" * 30)

    for rank in range(top_indices.shape[1]):
        class_index = top_indices[0, rank].item()
        probability = top_probabilities[0, rank].item()

        print(
            f"{rank + 1}. {CLASS_MAPPING[class_index]}: "
            f"{probability * 100:.2f}%"
        )


if __name__ == "__main__":
    main()