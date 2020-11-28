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
        default="region2lr1.0e-03_enc[8, 16, 8]_hidden10_ode[4, 8, 8, 4]_dec[8, 16, 8]_window128_epochs256_rtol0.0001_atol1e-06",
        type=str,
    )
    parser.add_argument(
        "--plot_indiv",
        default=False,
        type=bool,
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
    train_data_path = [
        pathlib.Path("data/-83.812_10.39_train.csv").resolve(),
        #pathlib.Path("data/73.125_18.8143_train.csv").resolve(),
        #pathlib.Path("data/126_7.5819_train.csv").resolve(),
    ]
    test_data_path = [
        pathlib.Path("data/-83.812_10.39_test.csv").resolve(),
        #pathlib.Path("data/73.125_18.8143_test.csv").resolve(),
        #pathlib.Path("data/126_7.5819_test.csv").resolve(),
    ]
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

    if not args.plot_indiv:
        # Get ready to plot all windows on a single image
        side_len = math.ceil(math.sqrt(num_windows))
        fig, axes = plt.subplots(side_len, side_len, figsize=(100, 35), sharey=True)
        plt.tight_layout()
        plt.suptitle(
            "Neural ODE: Predicted vs GT number of infections (extrapolations are to the RHS of the vertical line)",
            fontsize=40,
        )

    criterion = torch.nn.MSELoss()
    total_loss = 0

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
            infect_dist = torch.distributions.normal.Normal(
                infect_hat[GT_STEPS_FOR_EXTRAPOLATION:].squeeze(), 0.1
            )
            loss = -infect_dist.log_prob(gt_infect_window[GT_STEPS_FOR_EXTRAPOLATION:].squeeze()).mean()

            # Denormalize using means and stds from TRAINING data
            pred_infect = infect_hat * train_data.infect_stds + train_data.infect_means
            gt_infect = (
                gt_infect_window * test_data.infect_stds + test_data.infect_means
            )
            
            # Calculate MSE loss only for extrapolation
            extrapol_pred_infect = pred_infect[GT_STEPS_FOR_EXTRAPOLATION:]
            extrapol_gt_infect = gt_infect[GT_STEPS_FOR_EXTRAPOLATION:]

            #loss = criterion(extrapol_pred_infect, extrapol_gt_infect)
            total_loss += loss.item()

            # Plot predictions
            dates = test_data.dates[
                i * EXTRAPOLATION_WINDOW_LENGTH : (i + 1) * EXTRAPOLATION_WINDOW_LENGTH
            ].to_list()
            demarcation = dates[GT_STEPS_FOR_EXTRAPOLATION]
            pred_infect = pred_infect.squeeze(-1).squeeze(-1).numpy()
            gt_infect = gt_infect.squeeze(-1).squeeze(-1).numpy()

            # Plot each window individually
            if args.plot_indiv:
                first_date, last_date = dates[0].date(), dates[-1].date()
                plt.figure(figsize=(20, 10))
                plt.plot(dates, pred_infect, label="Prediction")
                plt.plot(dates, gt_infect, label="Ground truth")
                plt.axvline(x=demarcation, color="gray", linewidth=4, linestyle="solid")
                plt.xlabel("Date")
                plt.ylabel("Number of infections")
                plt.title(
                    "Neural ODE: Predicted vs GT number of infections (extrapolations are to the RHS of the vertical line)"
                )
                plt.legend(loc="best")
                individual_extrapolation_plot_filepath = (
                    plots_dir / f"{args.job_id}_{first_date}_{last_date}.png"
                )
                plt.savefig(individual_extrapolation_plot_filepath)
            else:
                # Prepare data to plot all windows on a single image
                row_idx = i // side_len
                col_idx = i % side_len
                axes[row_idx, col_idx].plot(dates, pred_infect, label="Prediction")
                axes[row_idx, col_idx].plot(dates, gt_infect, label="Ground truth")
                axes[row_idx, col_idx].axvline(
                    x=demarcation, color="gray", linewidth=2, linestyle="solid"
                )
                axes[row_idx, col_idx].set_xlabel("Date")
                axes[row_idx, col_idx].set_ylabel("num_infect")
            
    # Note down the test set loss
    loss_txt_filepath = plots_dir / f"{args.job_id}_test_loss.txt"
    avg_loss = total_loss / test_data.num_windows
    msg = f"Avg test loss: {avg_loss}\n"
    with open(loss_txt_filepath, "w") as f:
        f.writelines(msg)
    print(msg)

    # Plot all windows on a single image
    if not args.plot_indiv:
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
