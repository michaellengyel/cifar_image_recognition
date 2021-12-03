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

LABELS_PATH = "data/annotations/instances_val2017.json"


def main():

    with open(LABELS_PATH) as f:
        try:
            instance_labels_data = json.load(f)
        except JSONDecodeError:
            pass

    # Create dictionary structure with the image's id and file name from instance_labels_data["images"]
    images = {x["id"]: {"file_name": x["file_name"], "labels": []} for x in instance_labels_data["images"]}

    # Fill dictionary structure with bbox and category_id data from instance_labels_data["annotations"]
    for annotation in instance_labels_data["annotations"]:
        if annotation["image_id"] in images:
            images[annotation["image_id"]]["labels"].append({"category_id": annotation["category_id"], "bbox": annotation["bbox"]})

    # Dump to json
    out_file = open("bbox_labels.json", "w")
    json.dump(images, out_file, indent=6)
    out_file.close()


if __name__ == '__main__':
    main()
