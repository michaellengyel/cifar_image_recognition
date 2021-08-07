import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


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
    batch_size = 32

    filtered_data_path = "data/filtered_data.txt"
    load_model_path = "./models/model.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)

    dataset = MouseDataset(filtered_data_path, transform=None)

    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # MODEL
    model = RatNet(input_size, hidden_size, output_size)
    model.to(device)

    model.load_state_dict(torch.load(load_model_path, map_location=torch.device(device)))
    model.eval()

    with torch.no_grad():

        for batch_idx, (features, targets) in enumerate(train_loader):

            features = features.to(device=device)
            scores = model(features)

            scores_np = scores.cpu().numpy()  # Predictions of model during inference
            targets_np = targets.cpu().numpy()  # All points of the training data
            features_np = features.cpu().numpy()  # Start and End points of training data

            scores_np_x, scores_np_y = split_to_dims_all(scores_np[0])
            targets_np_x, targets_np_y = split_to_dims_all(targets_np[0])
            features_np_x, features_np_y = split_to_dims_all(features_np[0])

            plt.plot(scores_np_x, scores_np_y, '-ok')
            plt.plot(targets_np_x, targets_np_y, '-ob')
            plt.plot(features_np_x, features_np_y, 'or')
            plt.plot()
            plt.xlim([0, 500])
            plt.ylim([0, 500])
            plt.show()


if __name__ == '__main__':
    main()
