import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=64):
    # Define transformations for the training and test sets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the FashionMNIST dataset
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    val_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader