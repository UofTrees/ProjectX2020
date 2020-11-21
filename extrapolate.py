import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch

from projectx.data import Data



def extrapolate() -> None:

    # Retrieve the job_id
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", default="lr1.0e-03_enc[8, 16, 8]_hidden10_ode[16, 32, 32, 16]_dec[8, 16, 8]_window128_epochs256_rtol0.0001_atol1e-06", type=str)
    args = parser.parse_args()

    # Get all folders and files
    root = pathlib.Path("results").resolve()
    models_dir = root / "models"
    plots_dir = root / "plots"

    model_filepath = models_dir / f"{args.job_id}.pt"
    extrapolation_plot_filepath = plots_dir / f"{args.job_id}_extrapolation.png"
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
        window_length=200,
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

    # Extrapolate
    print("Extrapolation starts")
    preds = []
    ground_truth = []

    with torch.no_grad():
        time_window, gt_weather_window, gt_infect_window = next(data.windows())
        weather_window, infect_window = gt_weather_window[:100], gt_infect_window[:100]

        infect_hat = best_model(
                time_window=time_window,
                weather_window=weather_window,
                infect_window=infect_window,
            )

        pred_infect = infect_hat.squeeze(-1).squeeze(-1).numpy()
        gt_infect = gt_infect_window.squeeze(-1).squeeze(-1).numpy()

    #with torch.no_grad():
    #    time_window, gt_weather_window, gt_infect_window = next(data.windows())
    #    weather_window, infect_window = gt_weather_window[10:100], gt_infect_window[10:100]
    #    time_window = time_window[10:]

    #    infect_hat = best_model(
    #            time_window=time_window,
    #            weather_window=weather_window,
    #            infect_window=infect_window,
    #        )

    #    pred_infect = infect_hat.squeeze(-1).squeeze(-1).numpy()
    #    gt_infect = gt_infect_window[10:].squeeze(-1).squeeze(-1).numpy()


    x = np.arange(200)
    plt.figure(figsize=(20, 10))
    plt.plot(x, pred_infect, label="prediction")
    plt.plot(x, gt_infect, label="ground_truth")
    plt.xlabel("Step")
    plt.ylabel("num_infect")
    plt.title("Neural ODE: Prediction vs Ground Truth (the last 100 are extrapolation)")
    plt.legend(loc="best")
    plt.savefig(extrapolation_plot_filepath)

    print("Done")


if __name__ == "__main__":
    extrapolate()
