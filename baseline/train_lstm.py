import argparse
import pathlib

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from hyperparams import Hyperparameters
from lstm import MV_LSTM
from lstm_utils import split_sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--sequence_length", default=100, type=int)
    parser.add_argument("--num_epochs", default=5, type=int)
    parser.add_argument("--n_features", default=4, type=int)
    parser.add_argument("--n_hidden", default=20, type=int)

    return parser.parse_args()


def get_hyperparameters(args: argparse.Namespace) -> Hyperparameters:
    return Hyperparameters(
        lr=args.lr,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        num_epochs=args.num_epochs,
        n_features=args.n_features,
        n_hidden=args.n_hidden,
        variance=0.5,
    )


def train(
    path_train,
    path_valid,
    save_path,
    n_features,
    sequence_length,
    num_epochs,
    batch_size,
    lr,
    variance,
    n_hidden,
    metrics
):

    ## temporary
    #df2_train = pd.read_csv(pathlib.Path("../data/73.125_18.8143_train.csv").resolve())
    #del df2_train["date"]
    #sq2_train = df2_train.to_numpy()

    #df2_valid = pd.read_csv(pathlib.Path("../data/73.125_18.8143_valid.csv").resolve())
    #del df2_valid["date"]
    #sq2_valid = df2_valid.to_numpy()

    # Get the train and validation data
    df = pd.read_csv(path_train)
    del df["date"]
    sq = df.to_numpy()
    #sq = np.concatenate((sq, sq2_train), axis=0)
    X_train, y_train = split_sequences(sq, n_steps=sequence_length)

    df = pd.read_csv(path_valid)
    del df["date"]
    sq = df.to_numpy()
    #sq = np.concatenate((sq, sq2_valid), axis=0)
    X_valid, y_valid = split_sequences(sq, n_steps=sequence_length)


    # Get the model
    mv_net = MV_LSTM(n_features, sequence_length, n_hidden)
    optimizer = torch.optim.Adam(mv_net.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mv_net.to(device)

    mse = torch.nn.MSELoss()

    mv_net.train()
    best_valid_loss = None
    all_train_loss, all_val_loss = [], []
    num_batches_train = len(X_train) // batch_size
    num_batches_valid = len(X_valid) // batch_size

    # Train
    for t in range(num_epochs):
        train_loss = 0
        for b in range(0, len(X_train), batch_size):

            if b + batch_size > len(X_train):
                break

            inpt = X_train[b : b + batch_size, :, :]
            target = y_train[b : b + batch_size]

            if target.shape[0] != 0:
                x_batch = torch.from_numpy(inpt).float().to(device)
                y_batch = torch.from_numpy(target).float()

                mv_net.init_hidden(x_batch.size(0), device)
                output = mv_net(x_batch)

                if metrics == "MLE":
                    infect_dist = torch.distributions.normal.Normal(y_batch, variance)
                    loss = -infect_dist.log_prob(output.squeeze().cpu()).mean()
                elif metrics == "MSE":
                    loss = mse(output.cpu().view(-1), np.transpose(y_batch))

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item()

        # validation
        val_loss = 0
        with torch.no_grad():
            for b in range(0, len(X_valid), batch_size):

                if b + batch_size > len(X_valid):
                    break

                test_seq = X_valid[b : b + batch_size, :, :]
                label_seq = y_valid[b : b + batch_size]

                x_batch = torch.from_numpy(test_seq).float().to(device)
                y_batch = torch.from_numpy(label_seq).float()

                mv_net.init_hidden(x_batch.size(0), device)

                try:
                    output = mv_net(x_batch)
                    if metrics == "MLE":
                        infect_dist = torch.distributions.normal.Normal(y_batch, variance)
                        batch_val_loss = -infect_dist.log_prob(output.squeeze().cpu()).mean()
                    elif metrics == "MSE":
                        batch_val_loss = mse(output.cpu().view(-1), np.transpose(y_batch))

                    val_loss += batch_val_loss.item()
                except:
                    continue

        # compute train and validation loss per epoch
        train_loss = train_loss / num_batches_train
        valid_loss = val_loss / num_batches_valid

        all_train_loss.append(train_loss)
        all_val_loss.append(valid_loss)

        if best_valid_loss is None or valid_loss < best_valid_loss:
            torch.save(mv_net.state_dict(), save_path)
            best_valid_loss = valid_loss

        print("step : ", t, "training loss : ", train_loss)
        print("step : ", t, "validation loss : ", valid_loss)

    updates = [i for i in range(1, len(all_train_loss) + 1)]
    plt.plot(updates, all_train_loss, label="Training loss")
    plt.plot(updates, all_val_loss, label="Validation loss")
    plt.title("LSTM: loss curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./lstm_loss_curve.jpg")


if __name__ == "__main__":
    
    path_train = pathlib.Path("../data/-83.812_10.39_train.csv").resolve()
    path_valid = pathlib.Path("../data/-83.812_10.39_valid.csv").resolve()
    save_path = pathlib.Path("lstm_state_dict.pt").resolve()

    hyperparams = get_hyperparameters(parse_args())
    
    train(
        path_train,
        path_valid,
        save_path,
        lr=hyperparams.lr,
        batch_size=hyperparams.batch_size,
        sequence_length=hyperparams.sequence_length,
        num_epochs=hyperparams.num_epochs,
        n_features=hyperparams.n_features,
        n_hidden=hyperparams.n_hidden,
        variance=hyperparams.variance,
        metrics="MLE"
        )