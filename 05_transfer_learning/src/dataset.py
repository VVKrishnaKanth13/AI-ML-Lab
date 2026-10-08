from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import TrainingConfig


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "data" / "caltech256_subset"

TRAIN_ROOT = DATASET_ROOT / "train"
VALID_ROOT = DATASET_ROOT / "valid"
TEST_ROOT = DATASET_ROOT / "test"


# --------------------------------------------------
# Configuration
# --------------------------------------------------

config = TrainingConfig()


# --------------------------------------------------
# ImageNet preprocessing
# --------------------------------------------------

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


valid_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# Datasets
# --------------------------------------------------

train_data = datasets.ImageFolder(
    root=TRAIN_ROOT,
    transform=train_transform
)

valid_data = datasets.ImageFolder(
    root=VALID_ROOT,
    transform=valid_test_transform
)

test_data = datasets.ImageFolder(
    root=TEST_ROOT,
    transform=valid_test_transform
)


# --------------------------------------------------
# DataLoaders
# --------------------------------------------------

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

test_loader = DataLoader(
    test_data,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers
)


# --------------------------------------------------
# Class information
# --------------------------------------------------

CLASS_MAPPING = {
    index: class_name
    for class_name, index in train_data.class_to_idx.items()
}


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    print(f"Training images   : {len(train_data)}")
    print(f"Validation images : {len(valid_data)}")
    print(f"Test images       : {len(test_data)}")

    print(f"\nClasses:")
    print(train_data.classes)

    print(f"\nClass mapping:")
    print(CLASS_MAPPING)

    images, labels = next(iter(train_loader))

    print(f"\nImages shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")