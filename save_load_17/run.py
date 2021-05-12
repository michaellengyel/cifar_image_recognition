import torch
import torch.nn as nn
from model import ConvNet

FILE = './models/model.pth'


def main():

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)

    model = ConvNet().to(device)

    try:
        model.load_state_dict(torch.load(FILE))
        print("Finished loading model.")
        model.eval()
    except IOError:
        print("Failed to load model. Model might not exist.")
        return

    print("Print Network Parameters:")
    for param in model.parameters():
        print(param)

    print("Print model state dict: ", model.state_dict())

    with torch.no_grad():
        print("Perform inference/testing here...")


if __name__ == '__main__':
    main()
