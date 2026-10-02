import torch
import torch.nn as nn
import torch.optim as optim

from model import FASHIONMNISTMLP
from dataset import get_data_loaders

def train_model(epochs = 10, batch_size=64, learning_rate=0.001):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_data_loaders(batch_size)

    model = FASHIONMNISTMLP().to(device)

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    train_accuracies = []

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images  = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total

        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return model, train_losses, train_accuracies

trained_model, train_losses, train_accuracies = train_model(
    epochs=10,
    batch_size=64,
    learning_rate=0.001
)

torch.save(
    trained_model.state_dict(),
    "../models/fashion_mnist_mlp.pth"
)

torch.save(
    {
        "losses": train_losses,
        "accuracies": train_accuracies
    },
    "../results/training_history.pth"
)

print("Model saved as fashion_mnist_mlp.pth")
print("Training history saved as training_history.pth")

