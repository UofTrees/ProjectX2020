import argparse
import pathlib

import numpy as np
import torch
import matplotlib.pyplot as plt

from hyperparams import Hyperparameters
from lstm import BaselineLSTM
from rnn import BaselineRNN
from utils import split_sequences, get_data, drop_and_inject_timediff, get_region_coords

GT_STEPS_FOR_EXTRAPOLATION = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--region", default="cr", type=str)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--seq_len", default=100, type=int)
    parser.add_argument("--num_epochs", default=1, type=int)
    parser.add_argument("--n_hidden", default=20, type=int)
    parser.add_argument("--model_name", default="lstm", type=str)

    return parser.parse_args()


def get_hyperparameters(args: argparse.Namespace) -> Hyperparameters:
    return Hyperparameters(
        region=args.region,
        lr=args.lr,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_gt=GT_STEPS_FOR_EXTRAPOLATION,
        num_epochs=args.num_epochs,
        n_hidden=args.n_hidden,
        model_name=args.model_name,
        input_size=5,
        variance=0.1,
    )


def get_job_id(hyperparams: Hyperparameters) -> str:
    return (
        f"{hyperparams.region}"
        + f"_{hyperparams.model_name}"
        + f"_lr{hyperparams.lr:.1e}"
        + f"_batch{hyperparams.batch_size}"
        + f"_seq{hyperparams.seq_len}"
        + f"_epochs{hyperparams.num_epochs}"
        + f"_hidden{hyperparams.n_hidden}"
    )


def train(hyperparams: Hyperparameters) -> None:

    if hyperparams.seq_len > GT_STEPS_FOR_EXTRAPOLATION or hyperparams.seq_len < 0:
        raise AssertionError("--seq_len must a positive integer no greater than 100")

    root = pathlib.Path("baseline_results").resolve()
    if not root.exists():
        root.mkdir()

    models_dir = root / "models"
    if not models_dir.exists():
        models_dir.mkdir()
    plots_dir = root / "plots"
    if not plots_dir.exists():
        plots_dir.mkdir()

    job_id = get_job_id(hyperparams)
    model_filepath = models_dir / f"{job_id}.pt"
    loss_plot_filepath = plots_dir / f"{job_id}_loss.png"

    # Get the data
    region_coords = get_region_coords(hyperparams.region)
    train_data_path, valid_data_path = [], []
    for coord in region_coords:
        train_data_path.append(pathlib.Path(f"../data/{coord}_train.csv").resolve())
        valid_data_path.append(pathlib.Path(f"../data/{coord}_valid.csv").resolve())

    train_data = get_data(train_data_path)
    X_train, y_train = split_sequences(train_data, n_steps=hyperparams.num_gt)

    valid_data = get_data(valid_data_path)
    X_valid, y_valid = split_sequences(valid_data, n_steps=hyperparams.num_gt)

    # Get the model
    if hyperparams.model_name.lower() == "lstm":
        model = BaselineLSTM(
            hyperparams.input_size, hyperparams.seq_len, hyperparams.n_hidden
        )
    elif hyperparams.model_name.lower() == "rnn":
        model = BaselineRNN(
            hyperparams.input_size, hyperparams.seq_len, hyperparams.n_hidden
        )
    else:
        raise AssertionError("model_name must be 'lstm' or 'rnn'")
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train
    model.train()
    batch_size = hyperparams.batch_size
    best_valid_loss = None
    all_train_loss, all_val_loss = [], []
    num_batches_train = len(X_train) // batch_size
    num_batches_valid = len(X_valid) // batch_size

    for epoch in range(hyperparams.num_epochs):
        train_loss = 0
        for b in range(0, len(X_train), batch_size):

            if b + batch_size > len(X_train):
                break

            inpt = X_train[b : b + batch_size, :, :]
            target = y_train[b : b + batch_size]

            if target.shape[0] != 0:
                x_batch = torch.from_numpy(inpt).float().to(device)
                y_batch = torch.from_numpy(target).float()

                x_batch = drop_and_inject_timediff(x_batch, model.seq_len)

                model.init_hidden(x_batch.size(0), device)
                output = model(x_batch)

                infect_dist = torch.distributions.normal.Normal(
                    y_batch, hyperparams.variance
                )
                loss = -infect_dist.log_prob(output.squeeze().cpu()).mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item()

        # validation
        valid_loss = 0
        with torch.no_grad():
            for b in range(0, len(X_valid), batch_size):

                if b + batch_size > len(X_valid):
                    break

                test_seq = X_valid[b : b + batch_size, :, :]
                label_seq = y_valid[b : b + batch_size]

                x_batch = torch.from_numpy(test_seq).float().to(device)
                y_batch = torch.from_numpy(label_seq).float()

                x_batch = drop_and_inject_timediff(x_batch, model.seq_len)

                model.init_hidden(x_batch.size(0), device)

                output = model(x_batch)
                infect_dist = torch.distributions.normal.Normal(
                    y_batch, hyperparams.variance
                )
                loss = -infect_dist.log_prob(output.squeeze().cpu()).mean()

                valid_loss += loss.item()

        # compute train and validation loss per epoch
        avg_train_loss = train_loss / num_batches_train
        avg_valid_loss = valid_loss / num_batches_valid

        all_train_loss.append(avg_train_loss)
        all_val_loss.append(avg_valid_loss)

        if best_valid_loss is None or avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            print(f"Saving model at epoch {epoch:02d}\n")
            torch.save(model, model_filepath)

        print(f"Epoch {epoch}: Train {avg_train_loss} | Valid {avg_valid_loss}")

    updates = range(hyperparams.num_epochs)
    plt.plot(updates, all_train_loss, label="Training loss")
    plt.plot(updates, all_val_loss, label="Validation loss")
    if hyperparams.model_name == "lstm":
        plt.title("LSTM: Loss curve")
    elif hyperparams.model_name == "rnn":
        plt.title("RNN: Loss curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(loss_plot_filepath)


if __name__ == "__main__":
    hyperparams = get_hyperparameters(parse_args())
    train(hyperparams)
