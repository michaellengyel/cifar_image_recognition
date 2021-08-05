import random

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

    src_file = "./data.txt"
    dest_file = "filtered_data.txt"

    # Read in data
    df = pd.read_csv(src_file, delimiter=';', header=None, index_col=False, names=list(range(100)))
    df.fillna(0, inplace=True)
    data = df.to_numpy()

    # Replace nan (zero) elements with last valid coordinates
    for line in data:

        used_elements = 0

        for index in range(len(line)):
            if line[index] == 0:
                used_elements = index
                break

        for index in range(used_elements, len(line)):

            if not used_elements == 0:
                if index % 2 == 1:
                    line[index] = line[used_elements-1]
                elif index % 2 == 0:
                    line[index] = line[used_elements-2]

    # Normalize Data (display width = 500, display height = 500)
    # data = data / 500

    df = pd.DataFrame(data)
    df.to_csv(dest_file, index=False, header=False, sep=';')
    print(df)

    # Visualization
    for n in range(300):

        features_np_x, features_np_y = split_to_dims_all(data[[n]])
        print(n)
        print(data[[n]])
        print(features_np_x)
        print(len(features_np_x))
        print(features_np_y)
        print(len(features_np_y))

        plt.plot(features_np_x, features_np_y, '-or')
        plt.plot()
        plt.xlim([0, 500])
        plt.ylim([0, 500])
        plt.show()


if __name__ == '__main__':
    main()
