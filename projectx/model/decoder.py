from typing import List, Tuple

import torch
from projectx.utils import pairwise


class Decoder(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        fc_dims: List[int],
        output_dim: int,
    ) -> None:
        super().__init__()

        self.fcs = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, fc_dims[0])]
            + [
                torch.nn.Linear(input_dim, output_dim)
                for input_dim, output_dim in pairwise(fc_dims)
            ]
            + [torch.nn.Linear(fc_dims[-1], fc_dims[-1] * 2)]
            + [torch.nn.Linear(fc_dims[-1] * 2, fc_dims[-1])]
            + [torch.nn.Linear(fc_dims[-1], output_dim)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for fc in self.fcs[:-3]:
            x = fc(x)
            x = torch.tanh(x)
        x = self.fcs[-3](x)
        x = self.fcs[-2](x)
        x = self.fcs[-1](x)
        return x
