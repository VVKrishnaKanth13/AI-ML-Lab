import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from dataset import val_loader, CLASS_MAPPING
from model import MonkeyCNN


# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model path
MODEL_PATH = "../models/best.pt"


# Create model
model = MonkeyCNN(num_classes=10).to(DEVICE)

# Load best model
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

criterion = nn.CrossEntropyLoss()

model.eval()

total_loss = 0.0
correct = 0
total = 0

# Store predictions and actual labels
all_predictions = []
all_labels = []


with torch.inference_mode():

    for images, labels in val_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Forward pass
        outputs = model(images)

        # Loss
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)

        # Predictions
        predictions = outputs.argmax(dim=1)

        # Accuracy
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        # Store predictions and labels
        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())


# Final metrics
evaluation_loss = total_loss / total
evaluation_accuracy = 100 * correct / total

print(f"Evaluation Loss: {evaluation_loss:.4f}")
print(f"Evaluation Accuracy: {evaluation_accuracy:.2f}%")
print(f"Correct Predictions: {correct}/{total}")


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions
)

class_names = [
    CLASS_MAPPING[i]
    for i in range(10)
]


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(12, 10))

disp.plot(
    ax=ax,
    xticks_rotation=45
)

plt.title("Monkey Species - Confusion Matrix")
plt.tight_layout()

plt.show()