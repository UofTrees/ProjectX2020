import datetime
import pathlib
from typing import ClassVar, Generator, Tuple

import pandas as pd
import torch


class Data:
    _DATETIME_FORMAT: ClassVar[str] = "%Y-%m-%d %H:%M:%S"
    _TIMESTEP_DELTA: ClassVar[float] = 6.0

    _device: torch.device
    _window_length: int
    _batch_size: int

    _data: pd.DataFrame

    _weather_tensor: torch.Tensor
    _normalized_weather_tensor: torch.Tensor
    _weather_means: torch.Tensor
    _weather_stds: torch.Tensor

    _infect_tensor: torch.Tensor
    _normalized_infect_tensor: torch.Tensor
    _infect_means: torch.Tensor
    _infect_stds: torch.Tensor

    _time_tensor: torch.Tensor
    _normalized_time_tensor: torch.Tensor
    _normalized_timestep_delta: float

    def __init__(
        self,
        *,
        data_path: pathlib.Path,
        device: torch.device,
        window_length: int,
        batch_size: int,
    ) -> None:
        if not data_path.exists():
            raise ValueError(f"no such path {data_path}")

        self._device = device
        self._window_length = window_length
        self._batch_size = batch_size

        self._data = pd.read_csv(data_path)
        self._data = self._parse_datetime(self._data)

        self._weather_tensor, self._infect_tensor = self._dataframe_to_tensors(
            self._data
        )
        self._time_tensor = self._generate_timesteps()

        (
            self._normalized_weather_tensor,
            self._weather_means,
            self._weather_stds,
        ) = self._normalize_data(self._weather_tensor)
        (
            self._normalized_infect_tensor,
            self._infect_means,
            self._infect_stds,
        ) = self._normalize_data(self._infect_tensor)

        max_time = self._time_tensor.max()
        self._normalized_time_tensor = self._time_tensor / max_time
        self._normalized_timestep_delta = self._TIMESTEP_DELTA / max_time

    @property
    def infect_means(self) -> float:
        return self._infect_means

    @property
    def infect_stds(self) -> float:
        return self._infect_stds

    @property
    def num_windows(self) -> int:
        return (
            self._normalized_time_tensor.shape[0] - self._window_length
        ) // self._window_length

    def windows(
        self,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
        yield from zip(
            self._time_windows(),
            self._weather_windows(),
            self._infect_windows(),
        )

    def weather_at_time(self, t: torch.Tensor) -> torch.Tensor:
        """
        `t` is a scalar tensor due to how `odeint` works

        Obtain estimated weather conditions at that time `t`
        This is done by linearly interpolating between 2 adjacent timepoints
        for which we have weather conditions
        """
        normalized_timestep_delta = self._normalized_timestep_delta.to(self._device)

        inbetween = (
            ((t % normalized_timestep_delta) / normalized_timestep_delta)
            .unsqueeze(1)
            .to(self._device)
        )
        left_index = (t // normalized_timestep_delta).long().to(self._device)

        if any(torch.isnan(t)):
            raise ValueError()
        if any(left_index < 0):
            left_index = torch.zeros_like(left_index)
        if any(left_index >= len(self._normalized_weather_tensor) - 2):
            left_index = torch.zeros_like(left_index) - 2

        right_index = left_index + 1

        left_weather = self._normalized_weather_tensor[left_index]
        right_weather = self._normalized_weather_tensor[right_index]

        return left_weather * (1 - inbetween) + right_weather * inbetween

    @classmethod
    def _strptime(cls, datetime_string: str) -> datetime.datetime:
        return datetime.datetime.strptime(datetime_string, cls._DATETIME_FORMAT)

    @classmethod
    def _parse_datetime(cls, data: pd.DataFrame) -> pd.DataFrame:
        data.date = data.date.map(cls._strptime)
        return data

    def _dataframe_to_tensors(
        self, data: pd.DataFrame
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert the data to tensors
        This doesn't include any timing information because all the data is equally spaced.
        """
        
        return (
            torch.Tensor(
                [
                    data.RH,
                    data.CM,
                    data.CT,
                ]
            ).T.to(self._device),
            torch.Tensor(
                [
                    data.num_infect,
                ]
            ).T.to(self._device),
        )

    def _generate_timesteps(self) -> torch.Tensor:
        return torch.Tensor(
            [i * self._TIMESTEP_DELTA for i, _ in enumerate(self._weather_tensor)]
        ).to(self._device)

    @staticmethod
    def _normalize_data(
        data: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        means = data.mean(dim=0, keepdim=True)
        stds = data.std(dim=0, keepdim=True)
        data = (data - means) / stds
        return data, means, stds

    @staticmethod
    def _data_windows(
        data: torch.Tensor, *, window_length: int, batch_size: int
    ) -> Generator[torch.Tensor, None, None]:
        """
        Send one batch of `batch_size` windows.
        Each window has `window_length` datapoints.

        Output shape is (window_length, batch_size, data.shape[1])
        """

        data_windows = []
        for i in range(0, data.shape[0] - window_length, window_length):
            window_slice = slice(i, i + window_length)
            data_windows.append(data[window_slice])
            if (i + 1) % batch_size == 0 or i == data.shape[0] - window_length:
                yield torch.stack(data_windows, dim=1)
                data_windows = []

    def _weather_windows(self) -> Generator[torch.Tensor, None, None]:
        yield from self._data_windows(
            self._normalized_weather_tensor,
            window_length=self._window_length,
            batch_size=self._batch_size,
        )

    def _infect_windows(self) -> Generator[torch.Tensor, None, None]:
        yield from self._data_windows(
            self._normalized_infect_tensor,
            window_length=self._window_length,
            batch_size=self._batch_size,
        )

    def _time_windows(self) -> Generator[torch.Tensor, None, None]:
        yield from self._data_windows(
            self._normalized_time_tensor,
            window_length=self._window_length,
            batch_size=self._batch_size,
        )
