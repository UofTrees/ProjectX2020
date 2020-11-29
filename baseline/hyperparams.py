from dataclasses import dataclass
from typing import List

@dataclass
class Hyperparameters:
    lr: float
    batch_size: int
    seq_len: int
    num_gt: int
    num_epochs: int
    n_hidden: int
    model_name: str
    input_size: int
    variance: float
