import random

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
import time


class RatNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RatNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


class MouseDataset(Dataset):
    def __init__(self, file_name, transform=None):
        self.df = pd.read_csv(file_name, delimiter=';', header=None,  index_col=False, names=list(range(100)), dtype=np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        size = self.df.iloc[[item]].count(axis=1)
        features = torch.tensor([[self.df.iloc[item, 0], self.df.iloc[item, 1], self.df.iloc[item, size - 2], self.df.iloc[item, size - 1]]], dtype=torch.float32)
        labels = np.array([self.df.iloc[item].fillna(0)])
        labels = torch.from_numpy(labels).float()
        return features, labels


def split_to_dims_all(data):
    x = []
    y = []
    for track in data:
        for index in range(len(track)):
            if index % 2 == 0:
                x.append(track[index])
            else:
                y.append(track[index])
    return x, y


def main():

    input_size = 4
    hidden_size = 80
    output_size = 100
    learning_rate = 0.0001
    batch_size = 32
    num_epochs = 300

    filtered_data_path = "data/filtered_data.txt"

    model_name = "basic_rat-{}".format(int(time.time()))
    save_model_path = "./models/" + model_name + ".pth"
    writer = SummaryWriter(log_dir='runs/{}'.format(model_name))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)

    dataset = MouseDataset(filtered_data_path, transform=None)

    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # MODEL
    model = RatNet(input_size, hidden_size, output_size)
    model.to(device)

    # LOSS AND OPTIMIZER
    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    avg_losses = []

    for epoch in range(num_epochs):

        losses = []

        for batch_idx, (data, targets) in enumerate(train_loader):

            # Send data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            # FORWARD PASS
            scores = model(data)

            # Reshape data
            targets = targets[-1, :, :].flatten()
            scores = scores[-1, :, :].flatten()

            loss = criterion(scores, targets)

            losses.append(loss.item())

            # BACKWARD PASS
            optimizer.zero_grad()
            loss.backward()

            # GRADIENT DECENT AKA. ADAM STEP
            optimizer.step()

        print(f'Cost at epoch {epoch} is {sum(losses)/len(losses)}')
        avg_losses.append(sum(losses)/len(losses))
        writer.add_scalar("Avg Loss/Batch", sum(losses)/len(losses), epoch)

    print("Finishing training...")

    print("Flushing SummaryWriter...")
    writer.flush()
    print("Closing SummaryWriter...")
    writer.close()
    print("Saving model..")
    torch.save(model.state_dict(), save_model_path)

    plt.plot(avg_losses)
    plt.show()


if __name__ == '__main__':
    main()
