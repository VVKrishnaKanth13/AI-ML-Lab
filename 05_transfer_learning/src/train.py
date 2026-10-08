import torch
import torch.nn as nn
import torch.optim as optim

from config import TrainingConfig, DEVICE
from dataset import train_loader, valid_loader, train_data, valid_data
from model import TransferLearningModel


def train_one_epoch(model, criterion, optimizer, device):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear old gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update trainable parameters
        optimizer.step()

        # Statistics
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

    with torch.no_grad():

        for images, labels in valid_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Statistics
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

    # Create model
    model = TransferLearningModel(
        num_classes=len(train_data.classes)
    )

    model = model.to(device)

    # Loss function
    criterion = nn.NLLLoss()

    # Optimizer
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum
    )

    best_valid_loss = float("inf")

    for epoch in range(config.num_epochs):

        train_loss, train_accuracy = train_one_epoch(
            model,
            criterion,
            optimizer,
            device
        )

        valid_loss, valid_accuracy = validate(
            model,
            criterion,
            device
        )

        print(
            f"Epoch [{epoch + 1:02d}/{config.num_epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy * 100:.2f}% | "
            f"Val Loss: {valid_loss:.4f} "
            f"Val Acc: {valid_accuracy * 100:.2f}%"
        )

        # Save the best model
        if valid_loss < best_valid_loss:

            best_valid_loss = valid_loss

            torch.save(
                model.state_dict(),
                "../models/best_model.pth"
            )

            print("  --> Best model saved")


if __name__ == "__main__":
    main()