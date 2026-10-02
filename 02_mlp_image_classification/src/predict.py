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


def predict():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load validation/test data
    _, val_loader = get_data_loaders(batch_size=1)

    # Create model
    model = FASHIONMNISTMLP().to(device)

    # Load trained weights
    model.load_state_dict(
        torch.load(
            "../models/fashion_mnist_mlp.pth",
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    # Get one image
    images, labels = next(iter(val_loader))

    image = images[0]
    actual_label = labels[0]

    # Add batch dimension
    input_image = image.unsqueeze(0).to(device)

    # Inference
    with torch.inference_mode():

        output = model(input_image)

        # Convert log-probabilities → probabilities
        probabilities = torch.exp(output)

        predicted_label = output.argmax(dim=1).item()

    # Get probabilities for this image
    probabilities = probabilities[0].cpu()

    print("Actual class   :", CLASS_NAMES[actual_label.item()])
    print("Predicted class:", CLASS_NAMES[predicted_label])

    print("\nClass probabilities:")

    for i, probability in enumerate(probabilities):

        print(
            f"{CLASS_NAMES[i]:15s}: "
            f"{probability.item() * 100:.2f}%"
        )

    # Display image
    display_image = image.squeeze()

    # Undo normalization
    display_image = display_image * 0.5 + 0.5

    plt.imshow(display_image, cmap="gray")

    plt.title(
        f"Actual: {CLASS_NAMES[actual_label.item()]}\n"
        f"Predicted: {CLASS_NAMES[predicted_label]}"
    )

    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    predict()