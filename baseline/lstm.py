import torch


class MV_LSTM(torch.nn.Module):
    def __init__(self, n_features, seq_length, n_hidden=20):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = n_hidden
        self.n_layers = 1  # number of stacked layers
        self.l_lstm = torch.nn.LSTM(
            input_size=n_features,
            hidden_size=self.n_hidden,
            num_layers=self.n_layers,
            batch_first=True,
        )
        self.l_linear = torch.nn.Linear(self.n_hidden * self.seq_len, 1)
        self.hidden = torch.Tensor()

    def init_hidden(self, batch_size, device):
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden).to(device)
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden).to(device)
        self.hidden = (hidden_state, cell_state)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        lstm_out, self.hidden = self.l_lstm(x)
        # lstm_out has shape (batch_size, seq_len, num_directions * hidden_size)

        # For following linear layer we want to keep batch_size dimension and merge rest
        # .contiguous() -> solves tensor compatibility error
        x = lstm_out.contiguous().view(batch_size, -1)
        return self.l_linear(x)
