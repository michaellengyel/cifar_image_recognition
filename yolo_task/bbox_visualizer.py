import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import time
import json
from json.decoder import JSONDecodeError
import torch
import os
import pandas as pd
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

LABELS_PATH = "bbox_labels_train.json"
IMAGE_PATH = "data/images/train2017/"

def main():

    with open(LABELS_PATH) as f:
        try:
            label_data = json.load(f)
        except JSONDecodeError:
            pass

    for key, value in label_data.items():

        file_name = value["file_name"]
        labels = value["labels"]

        image_path = os.path.join(IMAGE_PATH, file_name)
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        for label in labels:
            bbox = label["bbox"]
            category_name = label["category_name"]
            draw.rectangle(((bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3])), outline="red")
            draw.text((bbox[0], bbox[1]), category_name, fill="red")

        fig, ax = plt.subplots(1, figsize=(15, 15))
        ax.imshow(image)
        plt.show()


if __name__ == '__main__':
    main()
