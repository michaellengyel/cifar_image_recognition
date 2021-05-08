import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Hyper parameters
    input_size = 784  # 28x28
    hidden_size = 100
    num_classes = 10
    batch_size = 100
    learning_rate = 0.001
    num_epochs = 2

    # MNIST
    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    examples = iter(train_loader)
    samples, labels = examples.next()
    print("Shape of samples: ", samples.shape)
    print("Shape of labels: ", labels.shape)
    print("Structure of first image: ", samples[0][0])
    print("Structure of first batch's labels: ", labels)

    # Plot the data
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(samples[i][0], cmap='gray')
    plt.show()

    model = NeuralNet(input_size, hidden_size, num_classes).to(device)

    # TRAINING

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Reshape the [100, 1, 28, 28] to [100, 784]
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backwards
            optimizer.zero_grad()  # Empty the values in the gradiant attribute
            loss.backward()  # Perform back propagation
            optimizer.step()  # Update parameters step

            # Print the the loss
            if (i + 1) % 100 == 0:
                print(f"epoch {epoch+1} / {num_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item():.4}")

    # TESTING
    with torch.no_grad():  # Disable gradiant calculation
        n_samples = 0
        n_correct = 0
        for images, labels in test_loader:
            # Reshape the [100, 1, 28, 28] to [100, 784]
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            outputs = model(images)

            # value, index
            _, predictions = torch.max(outputs, 1)
            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f"accuracy = {acc}")


if __name__ == '__main__':
    main()
