import torch
import torchdiffeq
from projectx.data import Data
from projectx.hyperparams import Hyperparameters
from projectx.model.decoder import Decoder
from projectx.model.encoder import Encoder
from projectx.model.lstm import LSTMModel


class Model(torch.nn.Module):
    hyperparams: Hyperparameters
    device: torch.device
    encoder: torch.nn.Module
    decoder: torch.nn.Module
    lstm_model: torch.nn.Module
    # odefunc: torch.nn.Module
    _use_diff_time: bool

    def __init__(
        self,
        data: Data,
        hyperparams: Hyperparameters,
        device: torch.device,
        use_diff_time: bool = False,
    ) -> None:
        super().__init__()

        self.hyperparams = hyperparams
        self.device = device

        self.encoder = Encoder(
            input_dim=self.hyperparams.input_dims + int(use_diff_time),
            fc_dims=self.hyperparams.encoder_fc_dims,
            hidden_dim=self.hyperparams.hidden_dims,
            dropout_rate=self.hyperparams.dropout_rate,
        ).to(self.device)
        self.lstm_model = LSTMModel(
            hidden_dim=self.hyperparams.hidden_dims,
            output_dim=self.hyperparams.hidden_dims,
        ).to(self.device)
        # self.odefunc = ODEFunc(
        #     data=data,
        #     device=self.device,
        #     func_dim=self.hyperparams.hidden_dims,
        #     fc_dims=self.hyperparams.odefunc_fc_dims,
        # ).to(self.device)
        self.decoder = Decoder(
            input_dim=self.hyperparams.hidden_dims,
            fc_dims=self.hyperparams.decoder_fc_dims,
            output_dim=self.hyperparams.output_dims,
        ).to(self.device)

        self._use_diff_time = use_diff_time

    def forward(
        self,
        weather_window: torch.Tensor,
        infect_window: torch.Tensor,
        time_window: torch.Tensor,
    ) -> torch.Tensor:
        cut_time_window = time_window[: weather_window.shape[0]]
        if self._use_diff_time:
            diff_time_window = torch.cat(
                (
                    torch.zeros(cut_time_window.shape[0], 1).to(self.device),
                    cut_time_window[:, 1:] - cut_time_window[:, :-1],
                ),
                dim=1,
            ).unsqueeze(-1)
            data_window = torch.cat(
                (diff_time_window, weather_window, infect_window), dim=2
            )
        else:
            data_window = torch.cat((weather_window, infect_window), dim=2)
        reversed_data_window = data_window.flip(0)

        # We feed the data to the encoder reversed so it comes up with `h` corresponding
        # to the latent encoding of the first element in the sequence chronologically,
        # with information from the future. We integrate that through time.
        h_init = torch.randn(1, 1, self.hyperparams.hidden_dims).to(self.device)
        _, h = self.encoder(reversed_data_window, h_init)

        hs = [h]
        # XXX: this is kinda impossible to train because it is basically a
        # cut_time_window.shape[0]-times composition of self.lstm_model
        # XXX: maybe we can pass in difftime here as well? maybe also weather info
        for i in range(time_window.shape[0] - 1):
            hs.append(self.lstm_model(hs[i]).unsqueeze(0).unsqueeze(0))
        hs = torch.stack(hs).to(self.device)

        # # We squeeze the time and `h` to accommodate the batchless way that `odeint` works.
        # time_window = time_window.squeeze()
        # h = h.squeeze(dim=0)

        # # Treat time steps as starting from 0
        # start_time = time_window[0]
        # self.odefunc.start_time = start_time
        # time_window = time_window - start_time

        # # We integrate `h` through time for the relevant timesteps.
        # # This gives us a sequence of latent encodings corresponding to the time steps.
        # hs = torchdiffeq.odeint(
        #     self.odefunc,
        #     h,
        #     time_window,
        #     rtol=self.hyperparams.rtol,
        #     atol=self.hyperparams.atol,
        #     method="euler",
        # ).to(self.device)

        # Decode the hidden states integrated through time to the infections.
        return self.decoder(hs)
