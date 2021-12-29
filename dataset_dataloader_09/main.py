import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class WineDataset(Dataset):
    def __init__(self, transform=None):
        feature_label = np.loadtxt('./coco/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.feature = feature_label[:, 1:]
        self.label = feature_label[:, [0]]
        self.n_samples = feature_label.shape[0]

        self.transform = transform

    def __getitem__(self, index):
        sample = self.feature[index], self.label[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples


class ToTensorTransform:
    def __call__(self, sample):
        feature, label = sample
        return torch.from_numpy(feature), torch.from_numpy(label)


class MultiplicationTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        feature, label = sample
        feature = feature * self.factor
        label = label * self.factor
        return feature, label


class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted


def main():

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Hyper parameters
    num_epochs = 2
    batch_size = 4
    learning_rate = 0.1

    # Creating composed transform from multiple user transforms
    composed = torchvision.transforms.Compose([ToTensorTransform(), MultiplicationTransform(3)])

    # Load coco
    dataset = WineDataset(transform=composed)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    total_samples = len(dataset)
    n_iterations = math.ceil(total_samples / batch_size)
    print(total_samples, n_iterations)

    # Create Model
    # FYI (13 is the number of attributes of a feature)
    model = LogisticRegression(13).to(device)

    # Loss and Optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    # TODO: Put coco in [0-1] ranged, one hot encode the labels
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(data_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward and Backward pass, update weights
            y_predicted = model(inputs)
            loss = criterion(y_predicted, labels)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Set gradiants to zero
            optimizer.zero_grad()

            # Log info
            if (i+1) % 5 == 0:
                #print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')
                print(f'loss = {loss.item()}')


if __name__ == '__main__':
    main()
