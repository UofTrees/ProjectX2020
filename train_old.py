import argparse
import pathlib
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from projectx.data import Data
from projectx.hyperparams import Hyperparameters
from projectx.model import Model

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--encoder_fc_dims", nargs="+", default=[8, 16, 8], type=int)
    parser.add_argument("--hidden_dims", default=10, type=int)
    parser.add_argument("--odefunc_fc_dims", nargs="+", default=[4, 8, 8, 4], type=int)
    parser.add_argument("--decoder_fc_dims", nargs="+", default=[8, 16, 8], type=int)
    parser.add_argument("--window_length", default=128, type=int)
    parser.add_argument("--num_epochs", default=32, type=int)
    parser.add_argument("--rtol", default=1e-4, type=float)
    parser.add_argument("--atol", default=1e-6, type=float)

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
        variance=0.1,
        window_length=args.window_length,
        num_epochs=args.num_epochs,
        rtol=args.rtol,
        atol=args.atol,
    )


def get_job_id(hyperparams: Hyperparameters) -> str:
    return (
        f"lr{hyperparams.lr:.1e}"
        + f"_enc{hyperparams.encoder_fc_dims}"
        + f"_hidden{hyperparams.hidden_dims}"
        + f"_ode{hyperparams.odefunc_fc_dims}"
        + f"_dec{hyperparams.decoder_fc_dims}"
        + f"_window{hyperparams.window_length}"
        + f"_epochs{hyperparams.num_epochs}"
        + f"_rtol{hyperparams.rtol}"
        + f"_atol{hyperparams.atol}"
    )


def train() -> None:
    hyperparams = get_hyperparameters(parse_args())
    job_id = get_job_id(hyperparams)

    # Generate folders where to save results (logs, models and plots)
    root = pathlib.Path("results").resolve()
    if not root.exists():
        root.mkdir()

    logs_dir = root / "logs"
    if not logs_dir.exists():
        logs_dir.mkdir()
    models_dir = root / "models"
    if not models_dir.exists():
        models_dir.mkdir()
    plots_dir = root / "plots"
    if not plots_dir.exists():
        plots_dir.mkdir()

    job_filepath = logs_dir / f"{job_id}.txt"
    model_filepath = models_dir / f"{job_id}.pt"
    extrapolation_plot_filepath = plots_dir / f"{job_id}_extrapolation.png"
    loss_plot_filepath = plots_dir / f"{job_id}_loss.png"

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

    # Get the data and model
    data_path = pathlib.Path("data/-83.812_10.39.csv").resolve()

    data = Data(
        data_path=data_path,
        device=device,
        window_length=hyperparams.window_length,
        batch_size=1,
    )

    model = Model(data=data, hyperparams=hyperparams, device=device)

    # criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams.lr,
    )

    # Train
    log("Training starts")

    all_avg_loss = []
    num_windows = data.num_windows
    lowest_avg_loss: Optional[float] = None
    for epoch in range(hyperparams.num_epochs):
        loss_total = 0.0
        for i, (time_window, weather_window, infect_window) in enumerate(
            data.windows()
        ):
            optimizer.zero_grad()

            infect_mu = model(
                time_window=time_window,
                weather_window=weather_window,
                infect_window=infect_window,
            )
            infect_dist = torch.distributions.normal.Normal(
                infect_mu.squeeze(), hyperparams.variance
            )

            # loss = criterion(infect_hat.squeeze(), infect_window.squeeze())
            loss = -infect_dist.log_prob(infect_window.squeeze()).mean()

            print(
                f"{epoch:02d} ({i:03d}/{num_windows:03d}): {loss.item():>2.4f}",
                end="\r",
            )
            loss_total += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = loss_total / data.num_windows
        all_avg_loss.append(avg_loss)
        log(f"\nEpoch {epoch:02d}: {avg_loss:1.4f}")

        if lowest_avg_loss is None or avg_loss < lowest_avg_loss:
            lowest_avg_loss = avg_loss
            log(f"Saving model at epoch {epoch:02d}\n")
            torch.save(model, model_filepath)

    x = np.arange(hyperparams.num_epochs)
    plt.figure()
    plt.plot(x, all_avg_loss, label="loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Neural ODE: loss curve")
    plt.legend(loc="best")
    plt.savefig(loss_plot_filepath)

    # Extrapolate
    # We'll give the first 100 time steps for it to produce z_t0
    # We'll then ask it to predict those first 100 and extrapolate the next 150
    log("Extrapolation starts")
    best_model = torch.load(model_filepath)

    # Get data with another window length
    data = Data(
        data_path=data_path,
        device=device,
        window_length=250,
        batch_size=1,
    )

    with torch.no_grad():
        time_window, gt_weather_window, gt_infect_window = next(data.windows())
        weather_window, infect_window = gt_weather_window[:100], gt_infect_window[:100]

        infect_hat = best_model(
            time_window=time_window,
            weather_window=weather_window,
            infect_window=infect_window,
        )

        pred_infect = infect_hat * data.infect_stds + data.infect_means
        gt_infect = gt_infect_window * data.infect_stds + data.infect_means

        pred_infect = pred_infect.squeeze(-1).squeeze(-1).numpy()
        gt_infect = gt_infect.squeeze(-1).squeeze(-1).numpy()

    x = np.arange(250)
    plt.figure(figsize=(20, 10))
    plt.plot(x, pred_infect, label="prediction")
    plt.plot(x, gt_infect, label="ground_truth")
    plt.xlabel("Step")
    plt.ylabel("num_infect")
    plt.title("Neural ODE: Prediction vs Ground Truth (the last 100 are extrapolation)")
    plt.legend(loc="best")
    plt.savefig(extrapolation_plot_filepath)

    log("Done")


if __name__ == "__main__":
    train()
