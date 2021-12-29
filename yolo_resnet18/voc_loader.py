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


def render_batch(image_batch, label_batch, S, C, B):

    for batch_index in range(image_batch.shape[0]):

        image = image_batch[batch_index, ...]
        image = image.permute(1, 2, 0)  # Permute the axis of the tensor to become an image
        image = image.detach().numpy()

        label = label_batch[batch_index, ...]
        label_shape = label_batch[batch_index, ...].shape

        image = Image.fromarray((image * 255).astype(np.ubyte))
        draw = ImageDraw.Draw(image)

        cell_width = image.width / S
        cell_height = image.height / S

        for x in range(label_shape[0]):
            for y in range(label_shape[1]):

                if label[x, y, -10] > 0.5:
                    width = label[x, y, -7] * cell_width
                    height = label[x, y, -6] * cell_height
                    x_pos = label[x, y, -9] * cell_width + y * cell_width - width / 2
                    y_pos = label[x, y, -8] * cell_height + x * cell_height - height / 2
                    draw.rectangle(((x_pos, y_pos), (x_pos + width, y_pos + height)), outline="red")

                if label[x, y, -5] > 0.5:
                    width = label[x, y, -2] * cell_width
                    height = label[x, y, -1] * cell_height
                    x_pos = label[x, y, -4] * cell_width + y * cell_width - width / 2
                    y_pos = label[x, y, -3] * cell_height + x * cell_height - height / 2
                    draw.rectangle(((x_pos, y_pos), (x_pos + width, y_pos + height)), outline="blue")

        fig, ax = plt.subplots(1)
        ax.imshow(image)
        plt.show()


def main():

    IMG_DIR = "../datasets/voc/images"
    LABEL_DIR = "../datasets/voc/labels"
    MAPPING_FILE = "../datasets/voc/100examples.csv"
    BATCH_SIZE = 16
    NUM_WORKERS = 2
    PIN_MEMORY = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.0
    EPOCHS = 100
    TRAIN = False
    LOAD_MODEL_FILE = "retrained_grad_true.pth.tar"
    S = 7
    C = 20
    B = 2

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
    # normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = Compose([convert_transform, resize_transform])

    train_dataset = VOCDataset(MAPPING_FILE, S=S, C=C, B=B, transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=True)

    for image_batch, label_batch in train_loader:
        print(image_batch.shape)
        print(label_batch.shape)
        render_batch(image_batch, label_batch, S=S, C=C, B=B)


if __name__ == '__main__':
    main()