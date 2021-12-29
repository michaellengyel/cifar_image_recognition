import torch
import os
import pandas as pd
from PIL import Image

import os
import json

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None):
        self.annotation = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotation.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [float(x) if float(x) != int(float(x)) else int(x) for x in label.replace("\n", "").split()]
                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotation.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width_cell, height_cell = (width * self.S, height * self.S)

            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[i, j, 21:25] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return image, label_matrix


def render_batch(image_batch, label_batch):
    for batch_index in range(image_batch.shape[0]):

        image = image_batch[batch_index, ...]
        image = image.permute(1, 2, 0)  # Permute the axis of the tensor to become an image
        image = image.detach().numpy()

        label = label_batch[batch_index, ...]
        label_shape = label_batch[batch_index, ...].shape

        image = Image.fromarray((image * 255).astype(np.ubyte))
        draw = ImageDraw.Draw(image)

        for x in range(label_shape[0]):
            for y in range(label_shape[1]):

                torch.set_printoptions(profile=None, sci_mode=False)

                if label[x, y, -10] > 0.5:
                    # print(label[x, y, :])
                    width = label[x, y, -7] * 64
                    height = label[x, y, -6] * 64
                    x_pos = label[x, y, -9] * 64 + y * 64 - width / 2
                    y_pos = label[x, y, -8] * 64 + x * 64 - height / 2
                    draw.rectangle(((x_pos, y_pos), (x_pos + width, y_pos + height)), outline="red")
                if label[x, y, -5] > 0.5:
                    # print(label[x, y, :])
                    width = label[x, y, -2] * 64
                    height = label[x, y, -1] * 64
                    x_pos = label[x, y, -4] * 64 + y * 64 - width / 2
                    y_pos = label[x, y, -3] * 64 + x * 64 - height / 2
                    draw.rectangle(((x_pos, y_pos), (x_pos + width, y_pos + height)), outline="blue")

        fig, ax = plt.subplots(1)
        ax.imshow(image)
        plt.show()
