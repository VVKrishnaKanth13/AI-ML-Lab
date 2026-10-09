from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2 as transforms

from config import TrainingConfig, IMAGE_SIZE


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = (
        PROJECT_ROOT.parent
        / "04_cnn_image_classification"
        / "data"
        / "10_Monkey_Species"
)
TRAIN_ROOT = DATASET_ROOT / "training" / "training"
VALID_ROOT = DATASET_ROOT / "validation" / "validation"

config = TrainingConfig()

# Dataset-specific normalization values from the course
MEAN = [0.4368, 0.4336, 0.3294]
STD = [0.2457, 0.2413, 0.2447]

# Common preprocessing for training and validation
preprocess = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        antialias=True
    ),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
])

# Training transformations: augmentation + normalization
train_transform = transforms.Compose([
    preprocess,
    transforms.RandomHorizontalFlip(),
    transforms.RandomErasing(p=0.4),
    transforms.RandomApply([
        transforms.RandomAffine(
            degrees=(30, 70),
            translate=(0.1, 0.3),
            scale=(0.5, 0.75)
        )
    ], p=0.1),
    transforms.Normalize(mean=MEAN, std=STD),
])

# Validation: no random augmentation
valid_transform = transforms.Compose([
    preprocess,
    transforms.Normalize(mean=MEAN, std=STD),
])

train_data = datasets.ImageFolder(
    root=TRAIN_ROOT,
    transform=train_transform
)

valid_data = datasets.ImageFolder(
    root=VALID_ROOT,
    transform=valid_transform
)

train_loader = DataLoader(
    train_data,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers
)

valid_loader = DataLoader(
    valid_data,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers
)

CLASS_MAPPING = {
    index: class_name
    for class_name, index in train_data.class_to_idx.items()
}


if __name__ == "__main__":
    print(f"Training images  : {len(train_data)}")
    print(f"Validation images: {len(valid_data)}")
    print(f"Classes          : {train_data.classes}")
    print(f"Class mapping    : {CLASS_MAPPING}")

    images, labels = next(iter(train_loader))

    print(f"\nImage batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print(f"Image dtype      : {images.dtype}")
    print(f"Image device     : {images.device}")