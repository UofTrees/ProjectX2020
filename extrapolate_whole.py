import math
import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch

from projectx.data import Data


EXTRAPOLATION_WINDOW_LENGTH = 250
GT_STEPS_FOR_EXTRAPOLATION = 100


def extrapolate() -> None:

    # Retrieve the job_id
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job_id",
        default="lr1.0e-03_enc[8, 16, 8]_hidden32_ode[32, 64, 32]_dec[8, 16, 8]_window128_epochs256_rtol0.0001_atol1e-06",
        type=str,
    )
    args = parser.parse_args()

    # Get all folders and files
    root = pathlib.Path("results").resolve()
    models_dir = root / "models"
    plots_dir = root / "plots"

    model_filepath = models_dir / f"{args.job_id}.pt"

    # Get device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    # Get the data
    train_data_path = pathlib.Path("data/-83.812_10.39_train.csv").resolve()
    test_data_path = pathlib.Path("data/-83.812_10.39_test.csv").resolve()
    train_data = Data(
        data_path=train_data_path,
        device=device,
        window_length=EXTRAPOLATION_WINDOW_LENGTH,
        batch_size=1,
    )
    test_data = Data(
        data_path=test_data_path,
        device=device,
        window_length=EXTRAPOLATION_WINDOW_LENGTH,
        batch_size=1,
    )

    # Load the model
    # Need to send different parts of the model to the correct device
    best_model = torch.load(model_filepath, map_location=device)
    best_model.device = device
    best_model.encoder = best_model.encoder.to(device)
    best_model.decoder = best_model.decoder.to(device)
    best_model.odefunc = best_model.odefunc.to(device)
    best_model.odefunc.data = test_data
    best_model.odefunc.device = device

    # Extrapolate
    # We'll give the first 100 time steps for it to produce z_t0
    # We'll then ask it to predict those first 100 and extrapolate the next 150
    print("Extrapolation starts")
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

            pred_infect = pred_infect.squeeze(-1).squeeze(-1).numpy()
            gt_infect = gt_infect.squeeze(-1).squeeze(-1).numpy()

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

    print("Done")


if __name__ == "__main__":
    extrapolate()
