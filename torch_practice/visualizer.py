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


class MouseDataset(Dataset):
    def __init__(self, file_name, transform=None):
        self.df = pd.read_csv(file_name, delimiter=';', header=None, names=list(range(100)), dtype=np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        size = self.df.iloc[[item]].count(axis=1)
        features = torch.tensor([[self.df.iloc[item, 0], self.df.iloc[item, 1], self.df.iloc[item, size - 2], self.df.iloc[item, size - 1]]], dtype=torch.float32)
        labels = np.array([self.df.iloc[item].fillna(0)], dtype=np.float)

        for i in range(len(labels[0])):
            if labels[0][i] == 0 and i % 2 == 1:
                labels[0][i] = self.df.iloc[item, size - 1]
            elif labels[0][i] == 0 and i % 2 == 0:
                labels[0][i] = self.df.iloc[item, size - 2]

        labels = torch.from_numpy(labels)
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = MouseDataset("./data.txt", transform=None)

    # VISUALIZATION CODE
    for n in range(300):
        i = random.randint(0, dataset.__len__())

        features, labels = dataset.__getitem__(i)

        print("ID: ", i)
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
        plt.show()


if __name__ == '__main__':
    main()
