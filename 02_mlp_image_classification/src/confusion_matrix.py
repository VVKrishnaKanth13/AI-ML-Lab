import torch
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from model import FASHIONMNISTMLP
from dataset import get_data_loaders


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


def generate_confusion_matrix():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    _, val_loader = get_data_loaders(batch_size=64)

    model = FASHIONMNISTMLP().to(device)

    model.load_state_dict(
        torch.load(
            "../models/fashion_mnist_mlp.pth",
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.inference_mode():

        for images, labels in val_loader:

            images = images.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.tolist()
            )

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    print("Confusion Matrix:")
    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )

    display.plot(
        xticks_rotation=45
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    generate_confusion_matrix()