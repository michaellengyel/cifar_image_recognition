import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


def main():

    # 1. Preparing coco

    x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

    x = torch.from_numpy(x_numpy.astype(np.float32))
    y = torch.from_numpy(y_numpy.astype(np.float32))

    y = y.view(y.shape[0], 1)

    n_samples, n_features = x.shape

    # 2. Model

    input_size = n_features
    output_size = 1
    model = nn.Linear(input_size, output_size)

    # 3. Loss and Optimizer

    learning_rate = 0.01
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 4. Training Loop

    num_epochs = 100
    for epoch in range(num_epochs):
        # Forward Pass
        y_predicted = model(x)
        loss = criterion(y_predicted, y)

        # Backward Pass
        loss.backward()

        # Update
        optimizer.step()

        # Empty the gradients
        optimizer.zero_grad()

        if (epoch + 1) % 10 == 0:
            print(f'epoch: (epoch+1), loss = {loss.item():.4f}')

    # Plot
    predicted = model(x).detach().numpy()
    plt.plot(x_numpy, y_numpy, 'r.')
    plt.plot(x_numpy, predicted, 'b.')
    plt.show()



if __name__ == '__main__':
    main()
