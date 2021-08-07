import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


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


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper paramaters
    input_size = 784
    hidden_size = 100
    num_classes = 10
    num_epochs = 10
    batch_size = 100
    learning_rate = 0.001

    # MNIST
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(),
                                               download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    examples = iter(train_loader)
    samples, labels = examples.next()
    print(samples.shape, labels.shape)

    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(samples[i][0],cmap='gray')
    plt.show()

    model = NeuralNet(input_size, hidden_size, num_classes)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    n_total_steps = len(train_loader)

    # Loss log
    loss_log = []

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Reshape the images
            #images = images.reshape(-1, 28*28)

            # Push data and labels to device
            images = images.to(device)
            labels = labels.to(device)

            # Forward Pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if(i+1) % 100 == 0:
                print(f'epoch {epoch + 1} / {num_epochs}, step {i+1}/{n_total_steps}, loss {loss.item():.4}')
                loss_log.append(loss.item())

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            #images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model(images)

            # value, index
            value, prediction = torch.max(outputs, 1)
            n_samples += labels.shape[0]
            n_correct = (prediction == labels).sum().item()

        accuracy = 100.0 * n_correct / n_samples
        print(f'accuracy = {accuracy}')

    plt.plot(loss_log)
    plt.show()


if __name__ == '__main__':
    main()
