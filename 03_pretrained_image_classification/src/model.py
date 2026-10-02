import torch
import torchvision

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.DEFAULT
    )
    model.eval()

    return model

if __name__ == "__main__":
    model = load_model()
    print(model)