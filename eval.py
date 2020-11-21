import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch

from projectx.data import Data


def eval() -> None:

    # Retrieve the job_id
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", default="lr1.0e-03_enc[8, 16, 8]_hidden10_ode[16, 32, 32, 16]_dec[8, 16, 8]_window128_epochs256_rtol0.0001_atol1e-06", type=str)
    args = parser.parse_args()

    # Get all folders and files
    root = pathlib.Path("results").resolve()
    models_dir = root / "models"
    plots_dir = root / "plots"

    model_filepath = models_dir / f"{args.job_id}.pt"
    inference_plot_filepath = plots_dir / f"{args.job_id}_inference.png"
    data_path = pathlib.Path("data/-83.812_10.39.csv").resolve()

    # Get device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    # Get the data and model
    data = Data(
        data_path=data_path,
        device=device,
        window_length=128,
        batch_size=1,
    )
    # Need to send different parts of the model to the correct device
    best_model = torch.load(model_filepath, map_location=device)
    best_model.device = device
    best_model.encoder = best_model.encoder.to(device)
    best_model.decoder = best_model.decoder.to(device)
    best_model.odefunc = best_model.odefunc.to(device)
    best_model.odefunc.data = data
    best_model.odefunc.device = device

    # Evaluate
    print("Evaluation starts")
    preds = []
    ground_truth = []

    # We currently only look at the first 200 time steps
    # Need to discuss with Jesse regarding the evaluation strategy
    with torch.no_grad():
        for i, (time_window, weather_window, infect_window) in enumerate(
            data.windows()
        ):
            if i == 200:
                break

            infect_hat = best_model(
                time_window=time_window,
                weather_window=weather_window,
                infect_window=infect_window,
            )

            # Decode the hidden states integrated through time to the infections.

            # Don't normalize
            pred = infect_hat[1][0][0]  # * data.infect_stds + data.infect_means
            preds.append(pred)
            gt = infect_window[1][0][0]  # * data.infect_stds + data.infect_means
            ground_truth.append(gt)

    x = np.arange(200)
    plt.figure(figsize=(20, 10))
    plt.plot(x, preds[:200], label="prediction")
    plt.plot(x, ground_truth[:200], label="ground_truth")
    plt.xlabel("Step")
    plt.ylabel("num_infect")
    plt.title("Neural ODE: Prediction vs Ground Truth (first 200 timesteps)")
    plt.legend(loc="best")
    plt.savefig(inference_plot_filepath)

    print("Done")


if __name__ == "__main__":
    eval()
