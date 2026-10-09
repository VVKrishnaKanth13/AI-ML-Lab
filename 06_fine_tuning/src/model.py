import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


class FineTuningModel(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        # Load MobileNetV3 Small with ImageNet pretrained weights
        self.model = mobilenet_v3_small(weights="DEFAULT")

        # Freeze the first 10 feature layers
        for param in self.model.features[:10].parameters():
            param.requires_grad = False

        # Replace the original ImageNet classifier output layer
        # Original: 1024 input features -> 1000 ImageNet classes
        # New:      1024 input features -> 10 monkey species
        self.model.classifier[3] = nn.Linear(
            in_features=1024,
            out_features=num_classes
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":

    model = FineTuningModel(num_classes=10)

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen_params = total_params - trainable_params

    print(model)
    print(f"\nTotal parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters   : {frozen_params:,}")

    # Verify that the model accepts a batch of images
    model.eval()

    with torch.inference_mode():
        images = torch.randn(2, 3, 224, 224)
        outputs = model(images)

    print(f"Input shape         : {images.shape}")
    print(f"Output shape        : {outputs.shape}")