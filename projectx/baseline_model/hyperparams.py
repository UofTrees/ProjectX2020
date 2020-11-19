from dataclasses import dataclass
from typing import List

@dataclass
class Hyperparameters:
    lr: float
    batch_size: int
    hidden_dims: int
    variance: float
    window_length: int
    num_epochs: int
