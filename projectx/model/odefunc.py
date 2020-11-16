from typing import List

import torch
from projectx.data import Data
from projectx.utils import pairwise


class ODEFunc(torch.nn.Module):
    data: Data
    device: torch.device
    start_time: int

    def __init__(
        self,
        data: Data,
        device: torch.device,
        *,
        func_dim: int,
        fc_dims: List[int],
    ) -> None:
        super().__init__()

        self.data = data
        self.device = device

        input_dim = func_dim + self.data.weather_at_time(torch.Tensor([0]).to(self.device)).shape[1]
        self.fcs = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, fc_dims[0])]
            + [
                torch.nn.Linear(input_dim, output_dim)
                for input_dim, output_dim in pairwise(fc_dims)
            ]
            + [torch.nn.Linear(fc_dims[-1], func_dim)]
        )

        for module in self.fcs.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0, std=0.1)
                torch.nn.init.constant_(module.bias, val=0)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if x.dim() == 0:
            x = x.unsqueeze(0)

        t = t + self.start_time
        weather = self.data.weather_at_time(t)
        x = torch.cat([weather, x], dim=-1).to(self.device)

        for fc in self.fcs[:-1]:
            x = fc(x)
            x = torch.tanh(x)
        x = self.fcs[-1](x)
        return x
