import torch
import matplotlib.pyplot as plt


def plot_training_curves():

    history = torch.load(
        "../results/training_history.pth",
        weights_only=True
    )

    losses = history["losses"]
    accuracies = history["accuracies"]

    epochs = range(1, len(losses) + 1)

    # Loss curve
    plt.figure()

    plt.plot(
        epochs,
        losses,
        marker="o"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Epoch")

    plt.grid()
    plt.show()

    # Accuracy curve
    plt.figure()

    plt.plot(
        epochs,
        accuracies,
        marker="o"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy (%)")
    plt.title("Training Accuracy vs Epoch")

    plt.grid()
    plt.show()


if __name__ == "__main__":
    plot_training_curves()