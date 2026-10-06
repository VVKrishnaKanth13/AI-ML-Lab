import torch
import torch.nn as nn
from torch.optim import Adam

from dataset import train_loader, val_loader
from model import MonkeyCNN
from config import TrainingConfig


config = TrainingConfig()

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = MonkeyCNN(num_classes=10)
model = model.to(DEVICE)


# --------------------------------------------------
# Loss and optimizer
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = Adam(
    model.parameters(),
    lr=config.learning_rate
)


# --------------------------------------------------
# Track best model
# --------------------------------------------------

best_val_accuracy = 0.0


# --------------------------------------------------
# Training
# --------------------------------------------------

for epoch in range(config.num_epochs):

    # ==============================================
    # Training
    # ==============================================

    model.train()

    running_train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_train_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        train_total += labels.size(0)
        train_correct += (
                predictions == labels
        ).sum().item()

    train_loss = (
            running_train_loss / len(train_loader)
    )

    train_accuracy = (
            100 * train_correct / train_total
    )


    # ==============================================
    # Validation
    # ==============================================

    model.eval()

    running_val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_val_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            val_total += labels.size(0)

            val_correct += (
                    predictions == labels
            ).sum().item()

    val_loss = (
            running_val_loss / len(val_loader)
    )

    val_accuracy = (
            100 * val_correct / val_total
    )


    # ==============================================
    # Save best model
    # ==============================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "../models/best.pt"
        )

        print("  ✓ Best model saved")


    # ==============================================
    # Print results
    # ==============================================

    print(
        f"Epoch [{epoch + 1}/{config.num_epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )


print()
print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)