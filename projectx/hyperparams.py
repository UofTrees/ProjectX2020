from dataclasses import dataclass
from typing import List


@dataclass
class Hyperparameters:
    lr: float
    dropout_rate: float
    input_dims: int
    output_dims: int
    encoder_fc_dims: List[int]
    hidden_dims: int
    odefunc_fc_dims: List[int]
    decoder_fc_dims: List[int]
    variance: float
    window_length: int
    num_epochs: int
    rtol: float
    atol: float
