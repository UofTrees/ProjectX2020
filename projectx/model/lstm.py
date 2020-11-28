import torch


class LSTMModel(torch.nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()

        self.rnn = torch.nn.LSTM(
            input_size=4,
            hidden_size=hidden_dim,
            num_layers=1,
            bias=True,
            batch_first=False,
            bidirectional=False,
        )

        self.fcl = torch.nn.Sequential(torch.nn.Linear(hidden_dim, output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize all the parameters of this network
        """
        self.rnn.weight_hh_l0.data.fill_(0)

        def init_normal(mod):
            if isinstance(mod, torch.nn.Linear):
                torch.nn.init.uniform_(mod.weight)

        self.fcl.apply(init_normal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.rnn(x)  # Remember: y is simply h_t for each t
        out = y.squeeze()
        return out
