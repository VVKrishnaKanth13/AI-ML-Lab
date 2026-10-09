from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from config import TrainingConfig, DEVICE
from dataset import train_loader, valid_loader, train_data
from model import FineTuningModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def train_one_epoch(model, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def validate(model, criterion, device):
    model.eval()

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

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def main():
    config = TrainingConfig()

    device = torch.device(
        DEVICE if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    model = FineTuningModel(
        num_classes=len(train_data.classes)
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    # Optimize only parameters that are trainable.
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate
    )

    best_valid_loss = float("inf")

    for epoch in range(config.num_epochs):
        train_loss, train_accuracy = train_one_epoch(
            model, criterion, optimizer, device
        )

        valid_loss, valid_accuracy = validate(
            model, criterion, device
        )

        print(
            f"Epoch [{epoch + 1:02d}/{config.num_epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy * 100:.2f}% | "
            f"Val Loss: {valid_loss:.4f} "
            f"Val Acc: {valid_accuracy * 100:.2f}%"
        )

        # Save the checkpoint with the lowest validation loss.
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

            torch.save(
                model.state_dict(),
                MODEL_DIR / "best_model.pth"
            )

            print("  --> Best model saved")

    print(f"\nBest validation loss: {best_valid_loss:.4f}")


if __name__ == "__main__":
    main()