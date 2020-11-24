import argparse
import math
import pathlib
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from projectx.data import Data
from projectx.hyperparams import Hyperparameters
from projectx.model import Model

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
EXTRAPOLATION_WINDOW_LENGTH = 250
GT_STEPS_FOR_EXTRAPOLATION = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--dropout_rate", default=0.5, type=float)
    parser.add_argument("--encoder_fc_dims", nargs="+", default=[8, 16, 8], type=int)
    parser.add_argument("--hidden_dims", default=4, type=int)
    parser.add_argument("--odefunc_fc_dims", nargs="+", default=[32, 32], type=int)
    parser.add_argument("--decoder_fc_dims", nargs="+", default=[8, 16, 8], type=int)
    parser.add_argument("--window_length", default=128, type=int)
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument("--rtol", default=1e-4, type=float)
    parser.add_argument("--atol", default=1e-6, type=float)

    return parser.parse_args()


def get_hyperparameters(args: argparse.Namespace) -> Hyperparameters:
    return Hyperparameters(
        lr=args.lr,
        dropout_rate=args.dropout_rate,
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

    # Get the training and validation data
    train_data_path = pathlib.Path("data/-83.812_10.39_train.csv").resolve()
    train_data = Data(
        data_path=train_data_path,
        device=device,
        window_length=hyperparams.window_length,
        batch_size=1,
        shuffle_windows=True,
    )
    valid_data_path = pathlib.Path("data/-83.812_10.39_valid.csv").resolve()
    valid_data = Data(
        data_path=valid_data_path,
        device=device,
        window_length=EXTRAPOLATION_WINDOW_LENGTH,
        batch_size=1,
    )

    # Set up the model
    model = Model(data=train_data, hyperparams=hyperparams, device=device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams.lr,
    )

    ###########################
    # Train
    ###########################
    log("Training starts")

    all_train_avg_loss, all_valid_avg_loss = [], []
    lowest_valid_avg_loss: Optional[float] = None

    for epoch in range(hyperparams.num_epochs):
        train_total_loss = 0.0
        for i, (time_window, weather_window, infect_window) in enumerate(
            train_data.windows()
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

            train_loss = -infect_dist.log_prob(infect_window.squeeze()).mean()

            print(
                f"{epoch:02d} ({i:03d}/{train_data.num_windows:03d}): {train_loss.item():>2.4f}",
                end="\r",
            )
            train_total_loss += train_loss.item()

            train_loss.backward()
            optimizer.step()

        train_avg_loss = train_total_loss / train_data.num_windows
        all_train_avg_loss.append(train_avg_loss)
        log(f"\nEpoch {epoch:02d} training loss: {train_avg_loss:1.4f}")

        # Validate
        valid_total_loss = 0
        with torch.no_grad():
            # For every window, use the first 100 to produce the initial latent state
            # Then predict num_infect for those 100, as well as for 150 time steps in the future
            for i, (time_window, gt_weather_window, gt_infect_window) in enumerate(
                valid_data.windows()
            ):
                weather_window_beginning, infect_window_beginning = (
                    gt_weather_window[:GT_STEPS_FOR_EXTRAPOLATION],
                    gt_infect_window[:GT_STEPS_FOR_EXTRAPOLATION],
                )

                # Give the model the validation data so its ODEFunc can correctly fetch weather data for evaluation
                # Then immediately give the model back the training data for the next epoch
                model.odefunc.data = valid_data
                valid_infect_mu = model(
                    time_window=time_window,
                    weather_window=weather_window_beginning,
                    infect_window=infect_window_beginning,
                )
                model.odefunc.data = train_data
                valid_infect_dist = torch.distributions.normal.Normal(
                    valid_infect_mu.squeeze(), hyperparams.variance
                )

                valid_loss = -valid_infect_dist.log_prob(
                    gt_infect_window.squeeze()
                ).mean()
                valid_total_loss += valid_loss.item()
        valid_avg_loss = valid_total_loss / valid_data.num_windows
        all_valid_avg_loss.append(valid_avg_loss)
        log(f"Epoch {epoch:02d} validation loss: {valid_avg_loss:1.4f}")

        if lowest_valid_avg_loss is None or valid_avg_loss < lowest_valid_avg_loss:
            lowest_valid_avg_loss = valid_avg_loss
            log(f"Saving model at epoch {epoch:02d}\n")
            torch.save(model, model_filepath)

    epochs = np.arange(hyperparams.num_epochs)
    plt.figure()
    plt.plot(epochs, all_train_avg_loss, label="Training loss")
    plt.plot(epochs, all_valid_avg_loss, label="Validation loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Neural ODE: loss curve")
    plt.legend(loc="best")
    plt.savefig(loss_plot_filepath)

    ###########################
    # Extrapolate
    ###########################

    # Get data with the window length for extrapolation
    test_data_path = pathlib.Path("data/-83.812_10.39_test.csv").resolve()
    test_data = Data(
        data_path=test_data_path,
        device=device,
        window_length=EXTRAPOLATION_WINDOW_LENGTH,
        batch_size=1,
    )
    # Use the best model and give it the test data so ODEFunc can fetch the correct weather data
    best_model = torch.load(model_filepath)
    best_model.odefunc.data = test_data

    # You can try using forward through time and taking the feeding only the first 100 timepoints
    # to the model encoder as follows:
    # best_model.backwards_through_time = False
    # best_model(..., timestep_to_stop_encoder=100)

    log("Extrapolation starts")
    num_windows = test_data.num_windows
    side_len = math.ceil(math.sqrt(num_windows))
    fig, axes = plt.subplots(side_len, side_len, figsize=(100, 35), sharey=True)
    plt.tight_layout()
    plt.suptitle(
        "Neural ODE: Predicted vs GT number of infections (extrapolations are to the RHS of the vertical line)",
        fontsize=40,
    )

    with torch.no_grad():
        # For every window, use the first 100 to produce the initial latent state
        # Then predict num_infect for those 100, as well as for 150 time steps in the future
        for i, (time_window, gt_weather_window, gt_infect_window) in enumerate(
            test_data.windows()
        ):
            weather_window, infect_window = (
                gt_weather_window[:GT_STEPS_FOR_EXTRAPOLATION],
                gt_infect_window[:GT_STEPS_FOR_EXTRAPOLATION],
            )

            infect_hat = best_model(
                time_window=time_window,
                weather_window=weather_window,
                infect_window=infect_window,
            )

            # Denormalize using means and stds from TRAINING data
            pred_infect = infect_hat * train_data.infect_stds + train_data.infect_means
            gt_infect = (
                gt_infect_window * train_data.infect_stds + train_data.infect_means
            )

            pred_infect = pred_infect.squeeze(-1).squeeze(-1).cpu().numpy()
            gt_infect = gt_infect.squeeze(-1).squeeze(-1).cpu().numpy()

            # Plot predictions
            dates = test_data.dates[
                i * EXTRAPOLATION_WINDOW_LENGTH : (i + 1) * EXTRAPOLATION_WINDOW_LENGTH
            ].to_list()
            demarcation = dates[GT_STEPS_FOR_EXTRAPOLATION]

            row_idx = i // side_len
            col_idx = i % side_len
            axes[row_idx, col_idx].plot(dates, pred_infect, label="Prediction")
            axes[row_idx, col_idx].plot(dates, gt_infect, label="Ground truth")
            axes[row_idx, col_idx].axvline(
                x=demarcation, color="gray", linewidth=2, linestyle="solid"
            )
            axes[row_idx, col_idx].set_xlabel("Date")
            axes[row_idx, col_idx].set_ylabel("num_infect")

    for j in range(num_windows + 1, side_len ** 2):
        row_idx = j // side_len
        col_idx = j % side_len
        fig.delaxes(axes[row_idx][col_idx])

    row_idx_final = (num_windows - 1) // side_len
    col_idx_final = (num_windows - 1) % side_len
    lines, labels = axes[row_idx_final, col_idx_final].get_legend_handles_labels()
    fig.legend(lines, labels, fontsize=40, loc="upper left")
    extrapolation_plot_filepath = plots_dir / f"{args.job_id}.png"
    plt.savefig(extrapolation_plot_filepath)

    log("Done")


if __name__ == "__main__":
    train()
