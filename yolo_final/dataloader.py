
from torchvision.datasets import CocoDetection
import numpy as np
from PIL import Image
import os
import torch

IMAGE_SIZE = 416
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class CustomDataset(CocoDetection):

    def __getitem__(self, index):

        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        bboxes = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        # image = Image.open(os.path.join(self.root, path)).convert('RGB')
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        image = np.array(image)

        argumentations = self.transforms(image=image, bboxes=[[12, 23, 43, 34]])
        image = argumentations['image']
        bboxes = argumentations['bboxes']

        target = torch.tensor([0, 0, 0, 0])

        return image, target
