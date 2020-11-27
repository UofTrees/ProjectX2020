import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from hyperparams import Hyperparameters

# multivariate data preparation
from lstm import MV_LSTM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--sequence_length", default=128, type=int)
    parser.add_argument("--num_epochs", default=256, type=int)

    return parser.parse_args()


def get_hyperparameters(args: argparse.Namespace) -> Hyperparameters:
    return Hyperparameters(
        lr=args.lr,
        batch_size=args.batch_size,
        variance=0.1,
        sequence_length=args.sequence_length,
        num_epochs=args.num_epochs,
    )


def split_sequences(seq, n_steps):
    X, y = [], []
    for i in range(len(seq) - n_steps - 1):
        seq_x, seq_y = seq[i : (i + n_steps), :], seq[i + n_steps, 3]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def train(
    path_train,
    path_valid,
    save_path,
    n_features=4,
    n_timesteps=100,
    train_episodes=200,
    batch_size=256,
    lr=0.001,
    variance=0.5,
    n_hidden=20,
    metrics="MLE"
):

    df = pd.read_csv(path_train)
    del df["date"]
    sq = df.to_numpy()
    X_train, y_train = split_sequences(sq, n_steps=n_timesteps)

    df = pd.read_csv(path_valid)
    del df["date"]
    sq = df.to_numpy()
    X_valid, y_valid = split_sequences(sq, n_steps=n_timesteps)

    mv_net = MV_LSTM(n_features, n_timesteps, n_hidden)
    optimizer = torch.optim.Adam(mv_net.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mv_net.to(device)

    mv_net.train()
    best_loss = 10000000
    loss_plot = []
    val_loss_plot = []
    for t in range(train_episodes):
        step_loss = 0
        for b in range(0, len(X_train), batch_size):
            if b + batch_size > len(X_train):
                break
            inpt = X_train[
                b : b + batch_size, :, :
            ]  # /np.linalg.norm(X_train[b:b+batch_size,:,:])
            target = y_train[
                b : b + batch_size
            ]  # /np.linalg.norm(y_train[b:b+batch_size])
            if target.shape[0] != 0:
                x_batch = (
                    torch.from_numpy(inpt).float().to(device)
                )  # torch.tensor(inpt,dtype=torch.float32)
                y_batch = torch.from_numpy(
                    target
                ).float()  # torch.tensor(target,dtype=torch.float32)
                mv_net.init_hidden(x_batch.size(0), device)
                output = mv_net(x_batch)

                if metrics == "MLE":
                    infect_dist = torch.distributions.normal.Normal(y_batch, 0.5)
                    loss = -infect_dist.log_prob(output.squeeze().cpu()).mean()
                elif metrics == "MSE":
                    loss = criterion(output.cpu().view(-1), np.transpose(y_batch))

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                step_loss += loss.item()

        # validation
        val_loss = 0
        with torch.no_grad():
            for b in range(0, len(X_valid), batch_size):
                if b + batch_size > len(X_valid):
                    break
                test_seq = X_valid[b : b + batch_size, :, :]  # /np.linalg.norm(X_test)
                label_seq = y_valid[
                    b : b + batch_size
                ]  # /np.linalg.norm(y_test[b:b+batch_size])
                x_batch = torch.from_numpy(test_seq).float().to(device)
                y_batch = torch.from_numpy(label_seq).float()
                mv_net.init_hidden(x_batch.size(0), device)
                try:
                    output = mv_net(x_batch)
                    if metrics == "MLE":
                        infect_dist = torch.distributions.normal.Normal(y_batch, variance)
                        batch_val_loss = -infect_dist.log_prob(
                            output.squeeze().cpu()
                        ).mean()
                    elif metrics == "MSE":
                        batch_val_loss = criterion(output.cpu().view(-1), np.transpose(y_batch))

                    val_loss += batch_val_loss.item()
                except:
                    continue

        # compute train and validation loss per epoch
        num_batches_train = len(X_train) // batch_size
        num_batches_test = len(X_valid) // batch_size
        train_loss = step_loss / num_batches_train
        valid_loss = val_loss / num_batches_test
        loss_plot.append(train_loss)
        val_loss_plot.append(valid_loss)
        if valid_loss < best_loss:
            torch.save(mv_net.state_dict(), save_path)
            best_loss = valid_loss

        print("step : ", t, "training loss : ", train_loss)
        print("step : ", t, "validation loss : ", valid_loss)

    updates = [i for i in range(1, len(loss_plot) + 1)]
    plt.plot(updates, loss_plot, label="Training loss")
    plt.plot(updates, val_loss_plot, label="Validation loss")
    plt.title("LSTM: loss curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./lstm_loss_curve.jpg")
    # plt.show()


if __name__ == "__main__":
    # train(path_train="./toy.csv", path_valid ='', save_path="./lstm_state_dict.pt")
    # train(path_train="../../data/-83.812_10.39_train.csv", path_valid="../../data/-83.812_10.39_valid.csv", save_path="./lstm_state_dict.pt")
    train(path_train="./-83.812_10.39_train.csv", path_valid="./-83.812_10.39_valid.csv", save_path="./lstm_state_dict.pt")
