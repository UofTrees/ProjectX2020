import torch

class LSTMModel(torch.nn.Module):
    def __init__(self, n_features, seq_length):
        super(LSTMModel, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 20 # number of hidden state's features
        self.n_layers = 1 # number of LSTM layers (stacked)
        self.l_lstm = torch.nn.LSTM(input_size = self.n_features,
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers,
                                 batch_first = True)
        self.l_linear = torch.nn.Linear(self.n_hidden*self.seq_len, 1)


    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden).cuda()
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden).cuda()
        self.hidden = (hidden_state, cell_state)


    def forward(self, x):
        batch_size, seq_len, _  = x.size()
        lstm_out, self.hidden = self.l_lstm(x, self.hidden)
        x = lstm_out.contiguous().view(batch_size, -1)
        return self.l_linear(x)
