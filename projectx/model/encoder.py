from typing import List, Tuple

import torch
from projectx.utils import pairwise


class Encoder(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        fc_dims: List[int],
        hidden_dim: int,
    ) -> None:
        super(Encoder, self).__init__()

        self.fcs = torch.nn.ModuleList(
            (
                [torch.nn.Linear(input_dim, fc_dims[0])]
                + [torch.nn.Linear(i_dim, o_dim) for i_dim, o_dim in pairwise(fc_dims)]
            )
            if fc_dims
            else []
        )
        self.rnn = torch.nn.RNN(
            input_size=fc_dims[-1] if fc_dims else input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            nonlinearity="tanh",
            bias=True,
            batch_first=False,
            dropout=0,
            bidirectional=False,
        )

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_original_shape = x.shape  # (seq_len, batch_size, input_dim)
        x = x.view(x.shape[0] * x.shape[1], x.shape[2])
        for fc in self.fcs:
            x = fc(x)
            x = torch.tanh(x)
        x = x.view(
            x_original_shape[0], x_original_shape[1], x.shape[1]
        )  # (seq_len, batch_size, fc_dims[-1])

        # In the paper, they just give h to a single-layer recognition network (RNN).
        # Here, we also take x into account.
        y, h = self.rnn(x, h)  # Remember: y is simply h_t for each t
        return y, h
