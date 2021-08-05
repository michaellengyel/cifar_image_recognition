import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class LSTMPredictor(nn.Module):
    def __init__(self, n_hidden=51):
        super(LSTMPredictor, self).__init__()
        self.n_hidden = n_hidden
        # lstm1, lstm2, linear
        self.lstm1 = nn.LSTMCell(1, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1)

    def forward(self, x, future=0):
        outputs = []
        n_samples = x.size(0)

        h_t1 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t1 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        h_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)

        for input_t in x.split(1, dim=1):
            # N, 1
            h_t1, c_t1 = self.lstm1(input_t, (h_t1, c_t1))
            h_t2, c_t2 = self.lstm2(h_t1, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)

        for i in range(future):
            h_t1, c_t1 = self.lstm1(output, (h_t1, c_t1))
            h_t2, c_t2 = self.lstm2(h_t1, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        return outputs


def main():

    N = 100
    L = 1000
    T = 20

    x = np.empty((N, L), dtype=np.float32)
    x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
    y = np.sin(x / 1.0 / T).astype(np.float32)

    plt.figure(figsize=(10, 8))
    plt.title("Sine Wave")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot(np.arange(x.shape[1]), y[0, :], 'r', linewidth=2.0)
    plt.show()

    # y = 100x1000
    train_input = torch.from_numpy(y[3:, :-1])  # 97x999
    train_target = torch.from_numpy(y[3:, 1:])  # 97x999
    test_input = torch.from_numpy(y[:3, :-1])  # 3x999
    test_target = torch.from_numpy(y[:3, 1:])  # 3x999

    model = LSTMPredictor()
    criterion = nn.MSELoss()

    optimizer = optim.LBFGS(model.parameters(), lr=0.8)

    n_steps = 10

    for i in range(n_steps):
        print("Step:", i)

        def closure():
            optimizer.zero_grad()
            out = model(train_input)
            loss = criterion(out, train_target)
            print("loss", loss.item())
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            future = 1000
            pred = model(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print("test loss", loss.item())
            y = pred.detach().numpy()

        plt.figure(figsize=(10, 8))
        plt.title("Predicted Sine Wave")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        n = train_input.shape[1]  # 999

        def draw(y_i, color):
            plt.plot(np.arange(n), y_i[:n], color, linewidth=2.0)
            plt.plot(np.arange(n, n + future), y_i[n:], color + ":", linewidth=2.0)

        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')

        plt.savefig("predicts%d.pdf" % i)
        plt.close()


if __name__ == '__main__':
    main()
