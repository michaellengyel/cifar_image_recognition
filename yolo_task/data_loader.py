import os
import json

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, json_file, img_dir, label_dir, S=7, B=2, C=20, transform=None):
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
        image = Image.open(self.img_dir + self.annotation[item]["file_name"])
        label = (self.annotation[item]["file_name"], self.annotation[item]["labels"][0]["category_name"])

        if self.transform:
            image = self.transform(image)

        return image, label


def plot_batch(batch):
    for image_index in range(batch.shape[0]):
        image = batch[image_index, ...]
        image = image.permute(1, 2, 0)  # Permute the axis of the tensor to become an image
        image = image.detach().numpy()
        plot_image(image)


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
    BATCH_SIZE = 3
    NUM_WORKERS = 1
    PIN_MEMORY = True

    tensor_transform = transforms.Resize((224, 224))
    resize_transform = transforms.ToTensor()
    transform = transforms.Compose([tensor_transform, resize_transform])

    train_dataset = COCODataset("bbox_labels_train.json", transform=transform, img_dir=TRAIN_IMG_DIR, label_dir=LABEL_DIR)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False, drop_last=False)

    for image_batch, label_batch in train_loader:
        print(label_batch)
        plot_batch(image_batch)


if __name__ == '__main__':
    main()

