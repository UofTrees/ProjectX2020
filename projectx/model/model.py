from typing import Optional

import torch
import torchdiffeq
from projectx.data import Data
from projectx.hyperparams import Hyperparameters
from projectx.model.decoder import Decoder
from projectx.model.encoder import Encoder
from projectx.model.odefunc import ODEFunc


class Model(torch.nn.Module):
    hyperparams: Hyperparameters
    device: torch.device
    encoder: torch.nn.Module
    decoder: torch.nn.Module
    odefunc: torch.nn.Module

    backwards_through_time: bool

    def __init__(
        self,
        data: Data,
        hyperparams: Hyperparameters,
        device: torch.device,
        *,
        backwards_through_time: bool = True,
    ) -> None:
        super().__init__()

        self.hyperparams = hyperparams
        self.device = device

        self.encoder = Encoder(
            input_dim=self.hyperparams.input_dims,
            fc_dims=self.hyperparams.encoder_fc_dims,
            hidden_dim=self.hyperparams.hidden_dims,
            dropout_rate=self.hyperparams.dropout_rate,
        ).to(self.device)
        self.odefunc = ODEFunc(
            data=data,
            device=self.device,
            func_dim=self.hyperparams.hidden_dims,
            fc_dims=self.hyperparams.odefunc_fc_dims,
        ).to(self.device)
        self.decoder = Decoder(
            input_dim=self.hyperparams.hidden_dims,
            fc_dims=self.hyperparams.decoder_fc_dims,
            output_dim=self.hyperparams.output_dims,
        ).to(self.device)

        self.backwards_through_time = backwards_through_time

    def forward(
        self,
        weather_window: torch.Tensor,
        infect_window: torch.Tensor,
        time_window: torch.Tensor,
        *,
        encoder_timesteps: Optional[int] = None,
    ) -> torch.Tensor:
        data_window = torch.cat((weather_window, infect_window), dim=2)
        if self.backwards_through_time:
            data_window = data_window.flip(0)

        # We feed the data to the encoder reversed so it comes up with `h` corresponding
        # to the latent encoding of the first element in the sequence chronologically,
        # with information from the future. We integrate that through time.
        h_init = torch.randn(1, 1, self.hyperparams.hidden_dims).to(self.device)
        if encoder_timesteps is not None:
            if self.backwards_through_time:
                _, h = self.encoder(data_window[:encoder_timesteps], h_init)
            else:
                _, h = self.encoder(data_window[:-encoder_timesteps], h_init)
        else:
            _, h = self.encoder(data_window[:encoder_timesteps], h_init)

        # We squeeze the time and `h` to accommodate the batchless way that `odeint` works.
        time_window = time_window.squeeze()
        h = h.squeeze(dim=0)

        # Treat time steps as starting from 0
        start_time = (
            time_window[0] if not encoder_timesteps else time_window[encoder_timesteps]
        )
        self.odefunc.start_time = start_time
        time_window = time_window - start_time

        # We integrate `h` through time for the relevant timesteps.
        # This gives us a sequence of latent encodings corresponding to the time steps.
        hs = torchdiffeq.odeint(
            self.odefunc,
            h,
            time_window[encoder_timesteps:],
            rtol=self.hyperparams.rtol,
            atol=self.hyperparams.atol,
            method="euler",
        ).to(self.device)

        # Decode the hidden states integrated through time to the infections.
        return self.decoder(hs)
