import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2 as transforms

from config import TrainingConfig

config = TrainingConfig()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_ROOT = os.path.join(PROJECT_ROOT,"data","10_Monkey_Species")

TRAIN_ROOT = os.path.join(DATASET_ROOT, "training", "training")
VAL_ROOT = os.path.join(DATASET_ROOT, "validation", "validation")

MEAN = [0.4368, 0.4336, 0.3294]
STD = [0.2457, 0.2413, 0.2447]

IMAGE_SIZE = (224, 224)

preprocess = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, antialias=True),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True)
])

VAL_TRANSFORM = transforms.Compose([
    preprocess,
    transforms.Normalize(mean=MEAN, std=STD)
])

TRAIN_TRANSFORM = transforms.Compose([
    preprocess,
    transforms.RandomHorizontalFlip(),
    transforms.RandomErasing(p=0.4),
    transforms.RandomApply([
        transforms.RandomAffine(degrees=(30, 70),
                                translate=(0.1, 0.3),
                                scale=(0.5, 0.75))
    ],
        p=0.1),
    transforms.Normalize(mean=MEAN, std=STD)
])

train_data = datasets.ImageFolder(root=TRAIN_ROOT, transform=TRAIN_TRANSFORM)

val_data = datasets.ImageFolder(root=VAL_ROOT, transform=VAL_TRANSFORM)

train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

CLASS_MAPPING = {
    0: "mantled_howler",
    1: "patas_monkey",
    2: "bald_uakari",
    3: "japanese_macaque",
    4: "pygmy_marmoset",
    5: "white_headed_capuchin",
    6: "silvery_marmoset",
    7: "common_squirrel_monkey",
    8: "black_headed_night_monkey",
    9: "nilgiri_langur"
}

if __name__ == "__main__":
    print("Training images:", len(train_data))
    print("Validation images:", len(val_data))

    print("Classes:", train_data.classes)
    print("Class mapping:", train_data.class_to_idx)

    images, labels = next(iter(train_loader))

    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)