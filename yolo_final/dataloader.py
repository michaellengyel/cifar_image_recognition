
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
        labels = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']

        image = np.array(Image.open(os.path.join(self.root, path)).convert('RGB'))
        bboxes = [x['bbox'] for x in labels]
        category_ids = [x['category_id'] for x in labels]

        transforms = self.transforms(image=image, bboxes=bboxes)
        image = transforms['image']
        bboxes = transforms['bboxes']

        target = torch.zeros((100, 5))
        for i, (box, category_id) in enumerate(zip(bboxes, category_ids)):
            target[i, :] = torch.tensor([box[0], box[1], box[2], box[3], category_id])

        return image, target
