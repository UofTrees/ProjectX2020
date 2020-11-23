import torch


class MV_LSTM(torch.nn.Module):
    def __init__(self, n_features, seq_length):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 20  # number of hidden state's features
        self.n_layers = 1  # number of LSTM layers (stacked)
        self.l_lstm = torch.nn.LSTM(
            input_size=n_features,
            hidden_size=self.n_hidden,
            num_layers=self.n_layers,
            batch_first=True,
        )
        self.l_linear = torch.nn.Linear(self.n_hidden * self.seq_len, 1)

    def forward(self, x):
        # x = torch.from_numpy(x)
        batch_size, seq_len, _ = x.size()

        lstm_out, self.hidden = self.l_lstm(x)
        # lstm_out(with batch_first = True) is
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest
        # .contiguous() -> solves tensor compatibility error
        x = lstm_out.contiguous().view(batch_size, -1)
        return self.l_linear(x)
