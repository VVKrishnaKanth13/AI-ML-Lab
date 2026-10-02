import torch
import torch.nn as nn

from model import FASHIONMNISTMLP
from dataset import get_data_loaders


def evaluate_model(batch_size=64):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # We only need the validation/test loader
    _, val_loader = get_data_loaders(batch_size)

    # Create the model architecture
    model = FASHIONMNISTMLP().to(device)

    # Load the trained weights
    model.load_state_dict(
        torch.load(
            "../models/fashion_mnist_mlp.pth",
            map_location=device,
            weights_only=True
        )
    )

    # Evaluation mode
    model.eval()

    criterion = nn.NLLLoss(reduction="sum")

    total_loss = 0.0
    correct = 0
    total = 0

    # No gradients required during evaluation
    with torch.inference_mode():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            # Get predicted class
            predictions = outputs.argmax(dim=1)

            # Count correct predictions
            correct += (predictions == labels).sum().item()

            # Count total images
            total += labels.size(0)

    # Average loss per image
    avg_loss = total_loss / total

    # Accuracy
    accuracy = 100 * correct / total

    print(f"Evaluation Loss: {avg_loss:.4f}")
    print(f"Evaluation Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    evaluate_model()