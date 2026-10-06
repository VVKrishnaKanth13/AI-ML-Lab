
import torch
import torch.nn as nn
from torchvision.ops import Conv2dNormActivation


class MonkeyCNN(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=5
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),

            # Block 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),

            # Block 3
            Conv2dNormActivation(
                in_channels=128,
                out_channels=256,
                kernel_size=3
            ),

            Conv2dNormActivation(
                in_channels=256,
                out_channels=256,
                kernel_size=3
            ),

            nn.MaxPool2d(kernel_size=2),

            # Block 4
            Conv2dNormActivation(
                in_channels=256,
                out_channels=512,
                kernel_size=3
            ),

            nn.MaxPool2d(kernel_size=2),

            # Reduce spatial dimensions
            nn.AdaptiveAvgPool2d((3, 3))
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                512 * 3 * 3,
                256
            ),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(
                256,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x

if __name__ == "__main__":

    model = MonkeyCNN()

    print(model)

    x = torch.randn(1, 3, 224, 224)

    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)