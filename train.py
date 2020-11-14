import argparse
import pathlib
from typing import Optional

import torch

from projectx.data import Data
from projectx.hyperparams import Hyperparameters
from projectx.model import Model

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--encoder_fc_dims", nargs="+", default=[8], type=int)
    parser.add_argument("--hidden_dims", default=2, type=int)
    parser.add_argument("--odefunc_fc_dims", nargs="+", default=[4], type=int)
    parser.add_argument("--decoder_fc_dims", nargs="+", default=[8], type=int)
    parser.add_argument("--window_length", default=64, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_epochs", default=256, type=int)

    return parser.parse_args()


def get_hyperparameters(args: argparse.Namespace) -> Hyperparameters:
    return Hyperparameters(
        lr=args.lr,
        input_dims=4,
        output_dims=1,
        encoder_fc_dims=args.encoder_fc_dims,
        hidden_dims=args.hidden_dims,
        odefunc_fc_dims=args.odefunc_fc_dims,
        decoder_fc_dims=args.decoder_fc_dims,
        window_length=args.window_length,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
    )


def get_job_id(hyperparams: Hyperparameters) -> str:
    return (
        f"lr{hyperparams.lr:.1e}"
        + f"_enc{hyperparams.encoder_fc_dims}"
        + f"_hidden{hyperparams.hidden_dims}"
        + f"_ode{hyperparams.odefunc_fc_dims}"
        + f"_dec{hyperparams.decoder_fc_dims}"
        + f"_window{hyperparams.window_length}"
        + f"_batch{hyperparams.batch_size}"
        + f"_epochs{hyperparams.num_epochs}"
    )


def train() -> None:
    hyperparams = get_hyperparameters(parse_args())
    job_id = get_job_id(hyperparams)

    # TODO: autogen folders
    # Where to log and save models
    root = pathlib.Path("logs").resolve()
    if not root.exists():
        root.mkdir()

    logs_dir = root / "output"
    if not logs_dir.exists():
        logs_dir.mkdir()
    models_dir = root / "models"
    if not models_dir.exists():
        models_dir.mkdir()

    job_filepath = logs_dir / f"{job_id}.txt"
    model_filepath = models_dir / f"{job_id}.pt"

    def log(msg: str):
        with open(job_filepath, "a") as f:
            f.writelines(msg + "\n")
        print(msg)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        log(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        log("Running on CPU")

    data_path = pathlib.Path("data/toy.csv").resolve()

    data = Data(
        data_path=data_path,
        device=device,
        window_length=hyperparams.window_length,
        batch_size=hyperparams.batch_size,
    )

    model = Model(data=data, hyperparams=hyperparams, device=device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams.lr,
    )

    log("Starting...")

    num_windows = data.num_windows
    lowest_loss_total: Optional[float] = None
    for epoch in range(hyperparams.num_epochs):
        loss_total = 0.0
        for i, (time_window, weather_window, infect_window) in enumerate(
            data.windows()
        ):
            optimizer.zero_grad()
            infect_hat = model(
                time_window=time_window,
                weather_window=weather_window,
                infect_window=infect_window,
            )
            loss = criterion(infect_hat.squeeze(), infect_window.squeeze())

            print(
                f"{epoch:02d} ({i:03d}/{num_windows:03d}): {loss.item():>2.4f}",
                end="\r",
            )
            loss_total += loss.item()

            loss.backward()
            optimizer.step()

        log(f"\nEpoch {epoch:02d}: {loss_total:1.4f}")

        if lowest_loss_total is None or loss_total < lowest_loss_total:
            lowest_loss_total = loss_total
            log(f"Saving model at epoch {epoch:02d}\n")
            torch.save(model, model_filepath)


if __name__ == "__main__":
    train()
