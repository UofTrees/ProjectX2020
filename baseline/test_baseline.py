import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from lstm import BaselineLSTM
from rnn import BaselineRNN
from utils import split_sequences


def extrapolate(pt_path, path, n_features = 4,n_timesteps = 100, batch_size = 1, n_hidden=20):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    df = pd.read_csv(path)
    del df['date']
    sq = df.to_numpy()
    X, _ = split_sequences(sq, n_steps=n_timesteps)

    if model_name == 'lstm':
        model = BaselineLSTM(n_features, n_timesteps, n_hidden)
    elif model_name == 'rnn':
        model = BaselineRNN(n_features, n_timesteps, n_hidden)


    model.load_state_dict(torch.load(pt_path))
    model.to(device)

    preds, labels = [], []

    # for j in range(0, len(X)-(len(X)%250)-1, 250):
    for j in range(0, 250*21, 250):
        X_trapolate = X[j+100:j+250]
        test_seq = X[j:j+1]
        pred, label = [], []
        with torch.no_grad():
            for i in range(150):
                x_batch = torch.from_numpy(test_seq).float().to(device)
                model.init_hidden(x_batch.size(0), device)
                output = model(x_batch)
                t = output.cpu().view(-1).numpy()[0]

                # Produce output of the extrapolation
                pred.append(t)
                label.append(X_trapolate[i][0][3])

                # Update test seq
                np_to_add = X_trapolate[i][0]
                np_to_add[-1] = t

                arr = test_seq.tolist()
                del arr[0][0]
                arr[0].append(np_to_add)
                test_seq = np.array(arr)

        preds.append(pred)
        labels.append(label)

    # Visualize preds vs labels
    # Also calculate average MSE loss per window
    losses = 0
    for j in range(21):
        # computing MSE loss
        pred = preds[j]
        label = labels[j]
        losses += np.mean(np.square(np.array(pred)-np.array(label)))

        # plotting
        updates = [i for i in range(1, 151)]
        plt.figure(figsize=(20, 10))
        plt.plot(updates, pred, label="Prediction")
        plt.plot(updates, label, label="Groud Truth")
        plt.title("LSTM: Prediction vs Groud Truth")
        plt.xlabel("Step")
        plt.ylabel("num_infect")
        plt.legend()
        # plt.show()
        plt.savefig(f"./lstm_pred_vs_gt_{j}.jpg")
    print(f"LSTM: average MSE Loss per window: {losses / 21}")


if __name__ == "__main__":
    # eval(pt_path = '', path = '')
    extrapolate(pt_path="./lstm_state_dict.pt", path="./-83.812_10.39_test.csv")
