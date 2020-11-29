import numpy as np
import pandas as pd
import torch


def split_sequences(seq, n_steps):
    X, y = [], []
    highest_time_step = len(seq)

    for i in range(highest_time_step - n_steps - 1):
        seq_x, seq_y = seq[i : (i + n_steps), :], seq[i + n_steps, 3]

        # Add normalized time steps
        time_steps = np.arange(i, i+n_steps) / highest_time_step
        time_steps = np.expand_dims(time_steps, axis=1)
        seq_x = np.concatenate((seq_x, time_steps), axis=1)

        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def get_data(data_paths):
    """Retrieve data from all the datasets in `data_paths`"""
    data_list = []
    for data_path in data_paths:
        df = pd.read_csv(data_path)
        del df["date"]
        data_list.append(df)

    all_data = pd.concat(data_list, axis=0)
    return all_data.to_numpy()


def drop_and_inject_timediff(x_batch: torch.Tensor, seq_len: int) -> torch.Tensor:
    """
    Randomly drop some time steps without replacement in `x_batch`
    Also inject differences in adjacent time steps to replace the raw time steps
    """

    # Find which time steps to keep
    original_length = x_batch.shape[1]
    indexes_to_keep = sorted(np.random.choice(
        original_length, seq_len, replace=False
    ))
    smaller_x_batch = x_batch[:, indexes_to_keep, :]

    # For each batch, get the time difference between adjacent time steps
    remaining_time_steps = smaller_x_batch[:, :, 4]
    time_differences_from_2nd_step = remaining_time_steps[:, 1:] - remaining_time_steps[:, :-1]
    time_differences = torch.zeros_like(remaining_time_steps)
    time_differences[:, 1:] = time_differences_from_2nd_step

    # Replace the time steps by the differences in time steps
    smaller_x_batch[:, :, 4] = time_differences

    return smaller_x_batch