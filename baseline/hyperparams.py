from dataclasses import dataclass
from typing import List

@dataclass
class Hyperparameters:
    lr: float
    batch_size: int
    sequence_length: int
    num_epochs: int
    n_features: int
    n_hidden: int
    variance: float