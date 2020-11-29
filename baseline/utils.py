import numpy as np
import pandas as pd

def split_sequences(seq, n_steps):
    X, y = [], []
    for i in range(len(seq) - n_steps - 1):
        seq_x, seq_y = seq[i : (i + n_steps), :], seq[i + n_steps, 3]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def get_data(data_paths):
    data_list = []
    for data_path in data_paths:
        df = pd.read_csv(data_path)
        del df["date"]
        data_list.append(df)

    all_data = pd.concat(data_list, axis=0)
    return all_data.to_numpy()