from typing import List, Tuple

import torch
from mr_node.utils import pairwise


class Encoder(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        fc_dims: List[int],
        hidden_dim: int,
        dropout_rate: float
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
        self.rnn = torch.nn.LSTM(
            input_size=fc_dims[-1] if fc_dims else input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bias=True,
            batch_first=False,
            bidirectional=False,
        )
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        `x` is arranged sequentially backwards in time
        It will now be encoded and the RNN will find a latent representation for the initial state of each window
        """

        # Encode x
        x_original_shape = x.shape  # (window_length, batch_size, input_dim)
        x = x.view(x.shape[0] * x.shape[1], x.shape[2])
        for fc in self.fcs:
            x = fc(x)
            x = torch.tanh(x)

        x = x.view(
            x_original_shape[0], x_original_shape[1], x.shape[1]
        )  # (window_length, batch_size, fc_dims[-1])

        x = self.dropout(x)

        # Make the RNN consume `x`, which is backwards in time
        # The returned `h` contains a latent initial state for each window
        y, (h, _) = self.rnn(x)
        return y, h
