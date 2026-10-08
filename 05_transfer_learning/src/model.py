import torch
import torch.nn as nn
from torchvision import models


class TransferLearningModel(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        # Load pretrained ResNet-50
        self.model = models.resnet50(weights="DEFAULT")

        # Freeze the pretrained feature extractor
        for param in self.model.parameters():
            param.requires_grad = False

        # Get the number of inputs expected by ResNet's classifier
        fc_inputs = self.model.fc.in_features

        # Replace the original ImageNet classifier
        self.model.fc = nn.Sequential(
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":

    model = TransferLearningModel(num_classes=10)

    print(model)

    # Count parameters
    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"\nTotal parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")