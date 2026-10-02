import torch
from pathlib import Path
from PIL import Image

from model import load_model
from preprocess import get_preprocess_transform


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLASS_NAMES_FILE = PROJECT_ROOT / "imagenet_classes.txt"
IMAGE_DIR = PROJECT_ROOT / "images"


def load_class_names():
    with open(CLASS_NAMES_FILE, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f]

    return class_names


def predict_image(image_path, top_k=5):

    # Load model
    model = load_model()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)

    # Load preprocessing transform
    transform = get_preprocess_transform()

    # Load class names
    class_names = load_class_names()

    # Read image
    image = Image.open(image_path).convert("RGB")

    # Preprocess image
    image_tensor = transform(image).to(device)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Inference
    with torch.inference_mode():
        outputs = model(image_tensor)

    # Convert scores to probabilities
    probabilities = torch.softmax(outputs, dim=1)

    # Get top-k predictions
    top_probabilities, top_indices = torch.topk(
        probabilities,
        top_k,
        dim=1
    )

    # Decode class IDs into class names
    predictions = []

    for probability, class_index in zip(
            top_probabilities[0],
            top_indices[0]
    ):
        class_id = class_index.item()
        class_name = class_names[class_id]
        confidence = probability.item()

        predictions.append(
            (class_name, confidence)
        )

    return predictions


if __name__ == "__main__":

    image_path = IMAGE_DIR / "dog.jpg"

    predictions = predict_image(
        image_path,
        top_k=5
    )

    print("Top predictions:")

    for rank, (class_name, confidence) in enumerate(
            predictions,
            start=1
    ):
        print(
            f"{rank}. {class_name}: "
            f"{confidence * 100:.2f}%"
        )