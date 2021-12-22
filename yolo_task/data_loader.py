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


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, json_file, img_dir, label_dir, S, C, B, transform=None):
        with open(json_file) as f:
            json_file_data = json.load(f)
        self.annotation = json_file_data
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, item):
        image = Image.open(self.img_dir + self.annotation[item]["file_name"]).convert('RGB')
        labels = self.annotation[item]["labels"]

        bboxes = [[x["category_id"], x["bbox"][0], x["bbox"][1], x["bbox"][2], x["bbox"][3]] for x in labels]

        # Original image width
        image_width = image.width
        image_height = image.height

        # Apply image transformations
        if self.transform:
            image, fake_labels = self.transform(image, bboxes)

        # Dimensions of the resized (transformed) images
        image_shape = image.shape

        # Segment width and height
        s_width = image_shape[1] / self.S
        s_height = image_shape[2] / self.S

        label_tensor = torch.zeros((self.S, self.S, self.C + 5))

        for bbox in bboxes:

            # Original box labels
            category, x, y, width, height = bbox

            x_norm = x / image_width
            y_norm = y / image_height
            w_norm = width / image_width
            h_norm = height / image_height

            i, j = int(self.S * y_norm), int(self.S * x_norm)
            x_cell, y_cell = self.S * x_norm - j, self.S * y_norm - i
            w_cell, h_cell = (w_norm * self.S, h_norm * self.S)

            if label_tensor[i, j, self.C] == 0:
                label_tensor[i, j, category] = 1  # Set class label
                label_tensor[i, j, self.C] = 1  # Set bbox probability
                label_tensor[i, j, self.C:self.C+5] = torch.tensor([1.0, x_cell, y_cell, w_cell, h_cell])  # Set bbox probability

        return image, label_tensor


def plot_batch(image_batch):
    for image_index in range(image_batch.shape[0]):
        image = image_batch[image_index, ...]
        image = image.permute(1, 2, 0)  # Permute the axis of the tensor to become an image
        image = image.detach().numpy()
        plot_image(image)


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

                if label[x, y, -10] > 0.3:
                    print(label[x, y, :])
                    x_pos = label[x, y, -9] * 64 + y * 64
                    y_pos = label[x, y, -8] * 64 + x * 64
                    width = label[x, y, -7] * 64
                    height = label[x, y, -6] * 64
                    draw.rectangle(((x_pos, y_pos), (x_pos + width, y_pos + height)), outline="black")
                if label[x, y, -5] > 0.3:
                    print(label[x, y, :])
                    x_pos = label[x, y, -4] * 64 + y * 64
                    y_pos = label[x, y, -3] * 64 + x * 64
                    width = label[x, y, -2] * 64
                    height = label[x, y, -1] * 64
                    draw.rectangle(((x_pos, y_pos), (x_pos + width, y_pos + height)), outline="red")

        fig, ax = plt.subplots(1)
        ax.imshow(image)
        plt.show()


def plot_image(image):
    if torch.is_tensor(image):
        image = image.permute(1, 2, 0)  # Permute the axis of the tensor to become an image
        image = image.detach().numpy()
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    plt.show()


def main():

    TRAIN_IMG_DIR = "data/images/train2017/"
    TEST_IMG_DIR = "data/images/test2017/"
    LABEL_DIR = "bbox_labels_train.json"
    BATCH_SIZE = 16
    NUM_WORKERS = 2
    PIN_MEMORY = True

    class Compose(object):
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, image, bboxes):
            for t in self.transforms:
                image = t(image)
            return image, bboxes

    # Types of preprocessing transforms we want to apply
    convert_transform = transforms.ToTensor()
    resize_transform = transforms.Resize((448, 448))

    transform = Compose([convert_transform, resize_transform])

    train_dataset = COCODataset("bbox_labels_train.json", S=7, C=100, B=2, transform=transform, img_dir=TRAIN_IMG_DIR, label_dir=LABEL_DIR)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False, drop_last=False)

    for image_batch, label_batch in train_loader:
        print(image_batch.shape)
        print(label_batch.shape)
        render_batch(image_batch, label_batch)


if __name__ == '__main__':
    main()

