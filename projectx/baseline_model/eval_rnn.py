from rnn import RNNModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def split_sequences(seq, n_steps):
    X, y = [], []
    for i in range(len(seq) - n_steps - 1):
        seq_x, seq_y = seq[i:(i + n_steps), :], seq[i + n_steps+150, 3]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def extrapolate(pt_path, path, n_features = 4,n_timesteps = 100, batch_size = 1):
    # use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mv_net.to(device)

    df = pd.read_csv(path)
    del df['date']
    sq = df.to_numpy()
    X, y = split_sequences(sq, n_steps=n_timesteps)

    mv_net = RNNModel(n_features,n_timesteps)
    mv_net.load_state_dict(torch.load(pt_path))
    X_trapolate = X[100:250]
    test_seq = X[0:1]
    pred = []
    label = []
    with torch.no_grad():
        for i in range(150):
            x_batch = torch.from_numpy(test_seq).float().to(device)
            mv_net.init_hidden(x_batch.size(0), device)
            output = mv_net(x_batch)
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
    # Visualize preds vs labels
    updates = [i for i in range(1, 151)]
    plt.figure(figsize=(20, 10))
    plt.plot(updates, pred, label="Prediction")
    plt.plot(updates, label, label="Groud Truth")
    plt.title("Prediction vs Groud Truth (MLE extrapolated)")
    plt.xlabel("Steps")
    plt.ylabel("num_infect")
    plt.legend()
    plt.savefig('./MLE Extrapolation for RNN.jpg')
    plt.show()

def eval(pt_path, path, n_features = 4,n_timesteps = 100, batch_size = 256):
    # use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mv_net.to(device)

    # Predict on test set
    df = pd.read_csv(path)
    del df['date']
    sq = df.to_numpy()
    X, y = split_sequences(sq, n_steps=n_timesteps)

    mv_net = RNNModel(n_features, n_timesteps)
    criterion = torch.nn.MSELoss(reduction='sum')  # reduction='sum' created huge loss value
    optimizer = torch.optim.Adam(mv_net.parameters(), lr=0.001)

    mv_net.load_state_dict(torch.load(pt_path))
    with torch.no_grad():
        preds = []
        labels = []
        loss_items = []
        for b in range(0, len(X), batch_size):
            test_seq = X[b:b + batch_size, :, :]  # /np.linalg.norm(X)
            label_seq = y[b:b + batch_size]  # /np.linalg.norm(y[b:b+batch_size])
            x_batch = torch.from_numpy(test_seq).float().to(device)
            y_batch = torch.from_numpy(label_seq).float()
            mv_net.init_hidden(x_batch.size(0), device)
            try:
                output = mv_net(x_batch)
                # loss = criterion(output.view(-1), np.transpose(y_batch))
                preds.extend(output.cpu().view(-1).numpy().tolist())
                labels.extend(label_seq.tolist())
                # loss_items.append(loss.item())
            except:
                continue
    updates = [i for i in range(1, len(preds) + 1)]
    plt.figure(figsize=(20, 10))
    plt.plot(updates[:200], preds[:200], label="Prediction")
    plt.plot(updates[:200], labels[:200], label="Groud Truth")
    plt.title("Prediction vs Groud Truth (first 200 timesteps)")
    plt.xlabel("Steps")
    plt.ylabel("num_infect")
    plt.legend()
    plt.savefig('./MLE Prediction for RNN.jpg')
    plt.show()
if __name__ == "__main__":
    eval(pt_path = '')
