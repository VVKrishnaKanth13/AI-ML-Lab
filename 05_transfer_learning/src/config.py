from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 0.01
    num_epochs: int = 25
    num_workers: int = 5
    momentum: float = 0.9


IMAGE_SIZE = 224
NUM_CLASSES = 10

DEVICE = "cuda"