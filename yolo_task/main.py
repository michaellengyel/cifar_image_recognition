import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import time

from model import YoloV1


def main():

    model = YoloV1()
    input = torch.randn((20, 3, 448, 448))
    model.eval()
    output = model(input)


if __name__ == '__main__':
    main()
