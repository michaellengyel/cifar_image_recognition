
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

    def __init__(self, root: str, annFile, transform=None, target_transform=None, transforms=None, catagory=None):
        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = []

        if catagory is None:
            self.ids = list(sorted(self.coco.imgs.keys()))
        else:
            for key, imgToAnn in self.coco.imgToAnns.items():
                replAnn = []
                contains_cat = False
                for ann in imgToAnn:
                    if ann['category_id'] == catagory:
                        contains_cat = True
                        replAnn.append(ann)
                self.coco.imgToAnns[key] = replAnn
                if contains_cat:
                    self.ids.append(key)
        print()

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
