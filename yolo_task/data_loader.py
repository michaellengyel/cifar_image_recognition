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

        original_image_width = image.width
        original_image_height = image.height

        if self.transform:
            image, fake_labels = self.transform(image, bboxes)

        image_shape = image.shape

        segment_width = image_shape[1] / self.S
        segment_height = image_shape[2] / self.S

        label_tensor = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        for bbox in bboxes:
            cat = bbox[0]
            x = bbox[1]
            y = bbox[2]
            width = bbox[3]
            height = bbox[4]

            x_corrected = x / original_image_width * image_shape[1]
            y_corrected = y / original_image_height * image_shape[2]
            width_corrected = width / original_image_width * image_shape[1]
            height_corrected = height / original_image_height * image_shape[2]

            x_center = x_corrected + (width_corrected / 2)
            y_center = y_corrected + (height_corrected / 2)
            index_x = int(x_center / segment_width)
            index_y = int(y_center / segment_height)
            label_tensor[index_x, index_y, cat] = 1  # Set class label
            label_tensor[index_x, index_y, self.C] = 1  # Set bbox probability
            label_tensor[index_x, index_y, self.C:self.C+5] = torch.tensor([1.0, x_corrected, y_corrected, width_corrected, height_corrected])  # Set bbox probability

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
                draw.rectangle(((label[x, y, -9], label[x, y, -8]), (label[x, y, -9] + label[x, y, -7], label[x, y, -8] + label[x, y, -6])), outline="red")

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
        render_batch(image_batch, label_batch)


if __name__ == '__main__':
    main()

