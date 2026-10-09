from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 20
    num_workers: int = 5


IMAGE_SIZE = 224
NUM_CLASSES = 10
DEVICE = "cuda"