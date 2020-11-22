from lstm import MV_LSTM

def split_sequences(seq, n_steps):
    X, y = [], []
    for i in range(len(seq) - n_steps - 1):
        seq_x, seq_y = seq[i:(i + n_steps), :], seq[i + n_steps+150, 3]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def eval(pt_path, path, n_features = 4,n_timesteps = 100, batch_size = 256):
    # Predict on test set
    df = pd.read_csv(path)
    del df['date']
    sq = df.to_numpy()
    X, y = split_sequences(sq, n_steps=n_timesteps)

    mv_net = MV_LSTM(n_features, n_timesteps)
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
            x_batch = torch.from_numpy(test_seq).float().cuda()
            y_batch = torch.from_numpy(label_seq).float()
            mv_net.init_hidden(x_batch.size(0))
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
    plt.savefig('./MLE Prediction for LSTM.jpg')
    plt.show()
if __name__ == "__main__":
    eval(pt_path = '')
