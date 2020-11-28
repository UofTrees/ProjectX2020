import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from baseline.hyperparams import Hyperparameters
from baseline.rnn import RNNModel


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
        batch_size = args.batch_size,
        variance=0.1,
        sequence_length=args.sequence_length,
        num_epochs=args.num_epochs,
    )

def split_sequences(seq, n_steps):
    X, y = [], []
    for i in range(len(seq) - n_steps - 1):
        seq_x, seq_y = seq[i:(i + n_steps), :], seq[i + n_steps, 3]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def train(path_train, path_valid, path_test, save_path, path_train_more = None, path_valid_more = None, path_test_more= None, save_path, n_features = 4, n_timesteps = 100, train_episodes = 256, batch_size = 256, lr=0.001):

    df_train = pd.read_csv(path_train)
    df_valid = pd.read_csv(path_valid)
    df_test = pd.read_csv(path_test)

    del df_train['date']
    del df_valid['date']
    del df_test['date']

    if not path_train_more is None or path_valid_more is None or path_test_more is None:
        df_surplus_train = pd.read_csv(path_train_more)
        df_surplus_valid = pd.read_csv(path_valid_more)
        del df_surplus_train['date']
        del df_surplus_valid['date']
        df_train = pd.concat([df_train, df_surplus_train], axis=0)
        df_valid = pd.concat([df_valid, df_surplus_valid], axis=0)

    sq_train = df_train.to_numpy()
    sq_test = df_test.to_numpy()
    sq_valid = df_valid.to_numpy()

    X_train, y_train = split_sequences(sq_train, n_timesteps)
    X_valid, y_valid = split_sequences(sq_valid, n_timesteps)
    X_test, y_test = split_sequences(sq_test, n_timesteps)

    model = RNNModel(n_features, n_timesteps)
    #criterion = torch.nn.MSELoss(reduction='sum')  # reduction='sum' created huge loss value
    optimizer = torch.optim.Adam(mv_net.parameters(), lr=lr)

    # use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
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
                model.init_hidden(x_batch.size(0))
                output = model(x_batch)
                # loss = criterion(output.cpu().view(-1), np.transpose(y_batch))
                infect_dist = torch.distributions.normal.Normal(y_batch, 0.1)
                loss = -infect_dist.log_prob(output.squeeze().cpu()).mean()
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
                test_seq = X_valid[b : b + batch_size, :, :]  
                label_seq = y_valid[
                    b : b + batch_size
                ]  
                x_batch = torch.from_numpy(test_seq).float().to(device)
                y_batch = torch.from_numpy(label_seq).float()
                model.init_hidden(x_batch.size(0))
                try:
                    output = model(x_batch)
                    # batch_val_loss = criterion(output.cpu().view(-1), np.transpose(y_batch))
                    infect_dist = torch.distributions.normal.Normal(y_batch, 0.1)
                    batch_val_loss = -infect_dist.log_prob(
                        output.squeeze().cpu()
                    ).mean()
                    val_loss += batch_val_loss.item()
                except:
                    continue

    num_batches_train = len(X_train) // batch_size
    num_batches_test = len(X_valid) // batch_size
    train_loss = step_loss / num_batches_train
    valid_loss = val_loss / num_batches_test
    loss_plot.append(train_loss)
    val_loss_plot.append(valid_loss)
    if valid_loss < best_loss:
        torch.save(model.state_dict(), 'mle_rnn_state_dict_model.pt')
        best_loss = valid_loss

    print("step : ", t, "training loss : ", train_loss)
    print("step : ", t, "validation loss : ", valid_loss)

    updates = [i for i in range(1, len(loss_plot) + 1)]
    plt.plot(updates, loss_plot, label="Training loss")
    plt.plot(updates, val_loss_plot, label="Validation loss")
    plt.title("MLE Loss Curve (batch_size=256, lr=0.001)")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('./MLE Curve for RNN.jpg')
    plt.show()

if __name__ == "__main__":