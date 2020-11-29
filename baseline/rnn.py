import torch


class BaselineRNN(torch.nn.Module):
    def __init__(self, input_size, seq_length, n_hidden):
        super(BaselineRNN, self).__init__()
        self.input_size = input_size
        self.seq_len = seq_length
        self.n_hidden = n_hidden
        self.n_layers = 1

        self.l_rnn = torch.nn.RNN(input_size=input_size,
                                  hidden_size=self.n_hidden,
                                  num_layers=self.n_layers,
                                  batch_first=True)

        self.l_linear = torch.nn.Linear(self.n_hidden * self.seq_len, 1)
        self.hidden = torch.Tensor()

    def init_hidden(self, batch_size, device):
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden).to(device)
        self.hidden = hidden_state

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # rnn_out has shape (batch_size, seq_len, num_directions * hidden_size)
        rnn_out, self.hidden = self.l_rnn(x, self.hidden)

        # For following linear layer we want to keep batch_size dimension and merge rest
        # .contiguous() -> solves tensor compatibility error
        x = rnn_out.contiguous().view(batch_size, -1)
        return self.l_linear(x)
