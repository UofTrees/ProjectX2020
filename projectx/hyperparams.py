from dataclasses import dataclass
from typing import List


@dataclass
class Hyperparameters:
    lr: float
    input_dims: int
    output_dims: int
    encoder_fc_dims: List[int]
    hidden_dims: int
    odefunc_fc_dims: List[int]
    decoder_fc_dims: List[int]
    window_length: int
    batch_size: int
    num_epochs: int
