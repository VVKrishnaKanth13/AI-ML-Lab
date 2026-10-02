import torch
import matplotlib.pyplot as plt

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


def find_misclassified(num_images=12):

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

    misclassified_images = []
    actual_labels = []
    predicted_labels = []

    with torch.inference_mode():

        for images, labels in val_loader:

            images = images.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            incorrect = predictions != labels.to(device)

            for i in range(len(images)):

                if incorrect[i]:

                    misclassified_images.append(
                        images[i].cpu()
                    )

                    actual_labels.append(
                        labels[i].item()
                    )

                    predicted_labels.append(
                        predictions[i].item()
                    )

                    if len(misclassified_images) == num_images:
                        break

            if len(misclassified_images) == num_images:
                break

    # Display images
    plt.figure(figsize=(12, 8))

    for i in range(num_images):

        image = misclassified_images[i].squeeze()

        plt.subplot(3, 4, i + 1)

        plt.imshow(image, cmap="gray")

        plt.title(
            f"Actual: {CLASS_NAMES[actual_labels[i]]}\n"
            f"Pred: {CLASS_NAMES[predicted_labels[i]]}"
        )

        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    find_misclassified()