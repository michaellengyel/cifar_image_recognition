import torch
import torch.nn as nn
from model import ConvNet

FILE = './models/model.pth'


def main():

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)

    model = ConvNet().to(device)

    torch.save(model.state_dict(), FILE)
    print("Finished saving model.")


if __name__ == '__main__':
    main()
