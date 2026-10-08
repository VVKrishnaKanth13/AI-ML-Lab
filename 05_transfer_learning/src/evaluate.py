import torch
import torch.nn as nn

from config import DEVICE
from dataset import test_loader, test_data
from model import TransferLearningModel


def main():

    device = torch.device(
        DEVICE if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # Create model
    model = TransferLearningModel(
        num_classes=len(test_data.classes)
    )

    # Load best trained weights
    model.load_state_dict(
        torch.load(
            "../models/best_model.pth",
            map_location=device,
            weights_only=True
        )
    )

    model = model.to(device)
    model.eval()

    criterion = nn.NLLLoss()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    test_loss = running_loss / total
    test_accuracy = correct / total

    print("\nTest Results")
    print("-" * 30)
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_accuracy * 100:.2f}%")
    print(f"Correct       : {correct}/{total}")


if __name__ == "__main__":
    main()