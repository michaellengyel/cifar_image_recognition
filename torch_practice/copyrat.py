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


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        #self.l1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        #self.relu = nn.ReLU()
        #self.l2 = nn.Linear(in_features=hidden_size, out_features=num_classes)
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.max = nn.MaxPool2d(kernel_size=3, stride=1)
        self.fc = nn.Linear(1728, 1000)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(1000, num_classes)


    def forward(self, x):
        #out = self.l1(x)
        #out = self.relu(out)
        #out = self.l2(out)
        #return out
        out = self.conv1(x)
        out = self.relu(out)
        out = self.max(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu2(out)
        out = self.fc2(out)
        return out


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
        self.df = pd.read_csv(file_name, delimiter=';', header=None, names=list(range(100)), dtype=np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.df)


    """
    def __getitem__(self, item):
        size = self.df.iloc[[item]].count(axis=1)
        features = torch.tensor([[self.df.iloc[item, 0], self.df.iloc[item, 1], self.df.iloc[item, size-2], self.df.iloc[item, size-1]]], dtype=torch.float32)
        labels = torch.tensor([self.df.iloc[item].fillna(0)], dtype=torch.float32)
        return features, labels
    """

    def __getitem__(self, item):
        size = self.df.iloc[[item]].count(axis=1)
        features = torch.tensor([[self.df.iloc[item, 0], self.df.iloc[item, 1], self.df.iloc[item, size - 2], self.df.iloc[item, size - 1]]], dtype=torch.float32)
        labels = np.array([self.df.iloc[item].fillna(0)], dtype=np.float)

        for i in range(len(labels[0])):
            if labels[0][i] == 0 and i % 2 == 1:
                labels[0][i] = self.df.iloc[item, size - 1]
            elif labels[0][i] == 0 and i % 2 == 0:
                labels[0][i] = self.df.iloc[item, size - 2]

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
    hidden_size = 50
    output_size = 100
    learning_rate = 0.001
    batch_size = 16
    num_epochs = 50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = MouseDataset("./data.txt", transform=None)

    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # MODEL
    model = RatNet(input_size, hidden_size, output_size)
    model.to(device)

    # LOSS AND OPTIMIZER
    #criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    avg_losses = []

    for epoch in range(num_epochs):

        losses = []

        for batch_idx, (data, targets) in enumerate(train_loader):

            # Send data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)
            #print("Beep", batch_idx)

            # FORWARD PASS
            scores = model(data)

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


    print('Finish Training')

    with torch.no_grad():

        plt.plot(avg_losses)
        plt.show()

        for i in range(100):

            x1 = random.randint(0, 500)
            y1 = random.randint(0, 500)
            x2 = random.randint(0, 500)
            y2 = random.randint(0, 500)

            features = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
            labels = model(features)

            features_np = features.detach().numpy()
            labels_np = labels.detach().numpy()

            features_np_x, features_np_y = split_to_dims_all(features_np)
            labels_np_x, labels_np_y = split_to_dims_all(labels_np)

            plt.plot(labels_np_x, labels_np_y, '-ok')
            plt.plot(features_np_x, features_np_y, 'or')
            plt.plot()
            plt.xlim([0, 500])
            plt.ylim([0, 500])
            plt.show()

    """
    # VISUALIZATION CODE
    for i in range(300):
        features, labels = data.__getitem__(i)
        print(features)
        print(labels)

        features_np = features.detach().numpy()
        labels_np = labels.detach().numpy()

        features_np_x, features_np_y = split_to_dims_all(features_np)
        labels_np_x, labels_np_y = split_to_dims_all(labels_np)

        plt.plot(labels_np_x, labels_np_y, '-ok')
        plt.plot(features_np_x, features_np_y, 'or')
        plt.plot()
        plt.xlim([0, 500])
        plt.ylim([0, 500])
    """

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = MouseDataset(file_name="./data/data.txt", transform=None)

    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)
        print("Beep", batch_idx)
    """


if __name__ == '__main__':
    main()
