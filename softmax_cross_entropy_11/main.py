import torch
import torch.nn as nn
import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def cross_entropy_loss(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss


def main():

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Softmax personal
    x = np.array([2.0, 1.0, 0.1])
    print("Softmax personal: ", softmax(x))

    # Softmax torch
    x = torch.tensor([2.0, 1.0, 0.1])
    print("Softmax torch: ", torch.softmax(x, dim=0))

    # Cross Entropy personal
    y_pred = np.array([2.0, 1.0, 0.1])
    y_pred_softmax = softmax(y_pred)
    y_actual = np.array([0, 1, 0])
    cel = cross_entropy_loss(y_actual, y_pred_softmax)
    print("Cross Entropy Loss personal: ", cel)

    # Cross Entropy torch (Automatically does softmax, y should not be one-hot encoded)
    # Y = class labels (not one-hot), Y_pred = raw scores (logits and not softmax)
    loss = nn.CrossEntropyLoss()
    y_pred = torch.tensor([[2.0, 1.0, 0.1]])
    y_actual = torch.tensor([1])  # The [1] means [0, 1, 0] in one-hot
    cel = loss(y_pred, y_actual)
    print("Cross Entropy Loss torch: ", cel)

    # Calculating Cross Entropy with torch for multiple samples:
    loss = nn.CrossEntropyLoss()
    y_pred = torch.tensor([[2.3, 2.1, 0.2], [1.1, 3.2, 5.1], [1.1, 0.1, 2.1]])
    y_actual = torch.tensor([0, 0, 0])
    cel = loss(y_pred, y_actual)
    print(cel)

    # Trying to perform multiple sample cross entropy with personal code:
    # Why isn't it the same?
    x = np.array([2.3, 1.1, 1.1])
    x_softmaxed = softmax(x)
    print(x_softmaxed)
    print((-np.log(x_softmaxed)).sum())



if __name__ == '__main__':
    main()
