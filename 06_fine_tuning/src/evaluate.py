import torch
import torch.nn as nn

from config import DEVICE
from dataset import valid_loader, valid_data
from model import FineTuningModel


def main():
    device = torch.device(
        DEVICE if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    model = FineTuningModel(
        num_classes=len(valid_data.classes)
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

    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    valid_loss = running_loss / total
    valid_accuracy = correct / total

    print("\nValidation Results")
    print("-" * 30)
    print(f"Validation Loss     : {valid_loss:.4f}")
    print(f"Validation Accuracy : {valid_accuracy * 100:.2f}%")
    print(f"Correct             : {correct}/{total}")


if __name__ == "__main__":
    main()