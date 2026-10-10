
from pathlib import Path
import torch

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Model configuration
IMAGE_SIZE = 224
NUM_CLASSES = 21

# ImageNet normalization for the pretrained model
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Select GPU when available
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


if __name__ == "__main__":
    for directory in [IMAGE_DIR, MODEL_DIR, RESULTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Device       : {DEVICE}")
    print(f"Image size   : {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Classes      : {NUM_CLASSES}")
