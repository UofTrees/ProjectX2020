import torch


class RNNModel(torch.nn.Module):
    def __init__(self, n_features, seq_length):
        super(RNNModel, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 40  # number of hidden states
        self.n_layers = 1  # number of RNN layers (stacked)

        self.l_rnn = torch.nn.RNN(input_size=n_features,
                                  hidden_size=self.n_hidden,
                                  num_layers=self.n_layers,
                                  batch_first=True)
        # according to pytorch docs RNN output is
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = torch.nn.Linear(self.n_hidden * self.seq_len, 1)
        self.hidden = torch.Tensor()

    def init_hidden(self, batch_size, device):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden).to(device)
        self.hidden = hidden_state

    def forward(self, x):
        # x = torch.from_numpy(x)
        batch_size, seq_len, _ = x.size()

        rnn_out, self.hidden = self.l_rnn(x, self.hidden)
        # rnn_out(with batch_first = True) is
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest
        # .contiguous() -> solves tensor compatibility error
        x = rnn_out.contiguous().view(batch_size, -1)
        return self.l_linear(x)
