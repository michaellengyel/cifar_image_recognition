import os
import fiftyone as fo
import fiftyone.zoo as foz

import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab

IMAGES_DIR = '/home/ad.adasworks.com/peter.lengyel/PycharmProjects/ml_practice/datasets/coco/images/val2017/'
LABELS_DIR = '/home/ad.adasworks.com/peter.lengyel/PycharmProjects/ml_practice/datasets/coco/annotations/instances_val2017.json'

# Load COCO formatted dataset
coco_dataset = fo.Dataset.from_dir(dataset_type=fo.types.COCODetectionDataset, data_path=IMAGES_DIR, labels_path=LABELS_DIR, include_id=True, label_field="")

# Verify that the class list for our dataset was imported
print(coco_dataset.default_classes)  # ['airplane', 'apple', ...]
print(coco_dataset)
session = fo.launch_app(coco_dataset, port=5151)
