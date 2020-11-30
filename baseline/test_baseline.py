import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from lstm import BaselineLSTM
from rnn import BaselineRNN
from utils import split_sequences, get_data, drop_and_inject_timediff, get_region_coords


EXTRAPOLATION_WINDOW_LENGTH = 250
GT_STEPS_FOR_EXTRAPOLATION = 100
# NUM_WINDOWS = 21
NUM_INFECT_INDEX = 3


def test() -> None:

    # Retrieve the region and job_id
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="cr", type=str)
    parser.add_argument(
        "--job_id",
        default="cr_lstm_lr1.0e-03_batch256_seq100_epochs1_hidden20",
        type=str,
    )
    args = parser.parse_args()

    # Get all folders and files
    root = pathlib.Path("baseline_results").resolve()
    models_dir = root / "models"
    plots_dir = root / "plots"
    model_filepath = models_dir / f"{args.job_id}.pt"

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the data
    region_coords = get_region_coords(args.region)
    test_data_path = []
    for coord in region_coords:
        test_data_path.append(pathlib.Path(f"../data/{coord}_test.csv").resolve())

    test_data = get_data(test_data_path)
    X_test, y_test = split_sequences(test_data, n_steps=GT_STEPS_FOR_EXTRAPOLATION)

    # Load the model to test
    model = torch.load(model_filepath, map_location=device)

    # set NUM_WINDOWS
    num_windows = len(X_test) // 250

    # Get all predictions and labels
    preds, labels = [], []
    for j in range(0, 250 * num_windows, 250):
        extrapolation_data = X_test[j + 100 : j + 250]
        test_seq = X_test[j : j + 1]

        pred, label = [], []
        with torch.no_grad():
            for i in range(150):
                x_batch = torch.from_numpy(test_seq).float().to(device)
                x_batch = drop_and_inject_timediff(x_batch, model.seq_len)
                model.init_hidden(x_batch.size(0), device)
                output = model(x_batch)
                t = output.cpu().view(-1).numpy()[0]

                # Produce output of the extrapolation
                pred.append(t)
                label.append(extrapolation_data[i][0][NUM_INFECT_INDEX])

                # Update test seq
                np_to_add = extrapolation_data[i][0]
                np_to_add[NUM_INFECT_INDEX] = t

                arr = test_seq.tolist()
                del arr[0][0]
                arr[0].append(np_to_add)
                test_seq = np.array(arr)

        preds.append(pred)
        labels.append(label)

    # Plot preds vs labels and find average MLE and MSE loss per window
    mse = torch.nn.MSELoss()
    total_mse_loss = 0
    total_mle_loss = 0

    for j in range(num_windows):
        pred = torch.Tensor(preds[j])
        label = torch.Tensor(labels[j])

        # Find losses and accumulate them
        mse_loss = mse(pred, label)
        infect_dist = torch.distributions.normal.Normal(pred, 0.5)
        mle_loss = -infect_dist.log_prob(label).mean()

        total_mle_loss += mle_loss.item()
        total_mse_loss += mse_loss.item()

        # plotting
        updates = range(150)
        plt.figure(figsize=(20, 10))
        plt.plot(updates, pred, label="Prediction")
        plt.plot(updates, label, label="Ground Truth")
        if "lstm" in args.job_id:
            plt.title("LSTM: Prediction vs Ground Truth")
        elif "rnn" in args.job_id:
            plt.title("RNN: Prediction vs Ground Truth")
        plt.xlabel("Step")
        plt.ylabel("num_infect")
        plt.legend()
        individual_extrapolation_plot_filepath = plots_dir / f"{args.job_id}_{j}.png"
        plt.savefig(individual_extrapolation_plot_filepath)

    # Note down the test set loss
    loss_txt_filepath = plots_dir / f"{args.job_id}_test_loss.txt"
    avg_mle_loss = total_mle_loss / num_windows
    avg_mse_loss = total_mse_loss / num_windows
    msg = f"Avg test MLE loss: {avg_mle_loss}\nAvg test MSE loss: {avg_mse_loss}\n"
    with open(loss_txt_filepath, "w") as f:
        f.writelines(msg)
        f.write("\n\n")
        for i in preds:
            f.write(str(i))
            f.write("\n\n")
    print(msg)


if __name__ == "__main__":
    test()
