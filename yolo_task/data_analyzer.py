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

LABELS_PATH = "data/annotations/person_keypoints_train2017.json"
IMAGE_PATH = "data/images/train2017/"


def main():

    with open(LABELS_PATH) as f:
        try:
            label_data = json.load(f)
        except JSONDecodeError:
            pass

    #for file_name, annotation in zip(label_data["images"], label_data["annotations"]):

    print(len(label_data["images"]))
    for file_name in label_data["images"]:

        bboxes = []
        image_path = os.path.join(IMAGE_PATH, file_name["file_name"])
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), "something123")

        print(len(label_data["annotations"]))
        for annotation in label_data["annotations"]:
            # print(file_name["file_name"], annotation["bbox"], file_name["id"])
            if file_name["id"] == annotation["image_id"]:
                bbox = annotation["bbox"]
                bboxes.append(annotation["bbox"])
                print(len(bboxes))
                draw.rectangle(((bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3])), outline="red")

        fig, ax = plt.subplots(1)
        ax.imshow(image)
        plt.show()


if __name__ == '__main__':
    main()
