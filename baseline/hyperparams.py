from dataclasses import dataclass
from typing import List

@dataclass
class Hyperparameters:
    lr: float
    batch_size: int
    sequence_length: int
    num_epochs: int
    n_hidden: int
    input_size: int
    variance: float
    model_name: str
