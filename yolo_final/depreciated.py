import os
from typing import Tuple, Any

from PIL import Image
import os
import os.path
from typing import Any, Callable, Optional, Tuple

import config

import torch
import torchvision.datasets.coco
import torchvision.transforms.transforms as transforms
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import numpy as np

# https://discuss.pytorch.org/t/how-to-efficiently-load-coco-dataset-to-dataloader/103380/5

# define pytorch transforms
transform1 = transforms.Compose([
     transforms.ToPILImage(),
     transforms.Resize((300, 300)),
     transforms.CenterCrop((100, 100)),
     transforms.RandomCrop((80, 80)),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomRotation(degrees=(-90, 90)),
     transforms.RandomVerticalFlip(p=0.5),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     ])

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class ToYolo(torch.nn.Module):

    def __init__(self, x):
        super(ToYolo, self).__init__()
        self.x = x

    def forward(self, target):
        """
        if len(target) > 0:
            first_bbox = torch.from_numpy(np.array(target[0]['bbox']))
        else:
            first_bbox = torch.tensor([0, 0, 0, 0])
        """
        first_bbox = torch.tensor([0, 0, 0, 0])
        return first_bbox

target_transform = transforms.Compose([
    ToYolo(3),
])


class CustomDataset(CocoDetection):
    def __init__(self, root, annFile, transform=None, transforms=None, target_transform=None):
        super().__init__(root, annFile, transform, transforms, target_transform)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        bboxes = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        # image = Image.open(os.path.join(self.root, path)).convert('RGB')
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        image = np.array(image)

        IMAGE_SIZE = 416
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        import cv2
        transforms = A.Compose(
            [
                A.LongestMaxSize(max_size=int(IMAGE_SIZE)),
                A.PadIfNeeded(min_height=int(IMAGE_SIZE), min_width=int(IMAGE_SIZE), border_mode=cv2.BORDER_CONSTANT),
                A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
                A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.6),
                A.ShiftScaleRotate(rotate_limit=10, p=0.4, border_mode=cv2.BORDER_CONSTANT),
                A.HorizontalFlip(p=0.5),
                A.Blur(p=0.2),
                A.CLAHE(p=0.2),
                A.Posterize(p=0.2),
                A.ToGray(p=0.1),
                ToTensorV2()
            ],
            bbox_params=A.BboxParams(format="coco", min_visibility=0.4, label_fields=[])
        )

        argumentations = transforms(image=image, bboxes=[[12, 23, 43, 34]])
        image = argumentations['image']
        bboxes = argumentations['bboxes']

        target = torch.tensor([0, 0, 0, 0])

        return image, target

    def __len__(self) -> int:
        return len(self.ids)


def main():

    root_train = "/home/ad.adasworks.com/peter.lengyel/PycharmProjects/ml_practice/datasets/coco/images/train2017/"
    annFile_train = "/home/ad.adasworks.com/peter.lengyel/PycharmProjects/ml_practice/datasets/coco/annotations/instances_train2017.json"
    root_val = "/home/ad.adasworks.com/peter.lengyel/PycharmProjects/ml_practice/datasets/coco/images/val2017/"
    annFile_val = "/home/ad.adasworks.com/peter.lengyel/PycharmProjects/ml_practice/datasets/coco/annotations/instances_val2017.json"

    # train_dataset = CocoDetection(root=root_train, annFile=annFile_train, transform=transform, target_transform=target_transform)
    # val_dataset = CocoDetection(root=root_val, annFile=annFile_val, transform=transform, target_transform=target_transform)
    # train_dataset = CocoDetection(root=root_train, annFile=annFile_train, transforms=config.transforms)
    # val_dataset = CocoDetection(root=root_val, annFile=annFile_val, transforms=config.transforms)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=16, num_workers=4, pin_memory=True, shuffle=False, drop_last=True)
    # val_loader = DataLoader(dataset=val_dataset, batch_size=16, num_workers=4, pin_memory=True, shuffle=False, drop_last=True)

    train_dataset = CustomDataset(root=root_train, annFile=annFile_train, transforms=config.transforms)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, num_workers=4, pin_memory=True, shuffle=False, drop_last=True)

    for index, (x, y) in enumerate(train_loader):
        print(index)
        print(x.shape)
        print(y)

    #for index, (x, y) in enumerate(val_loader):
    #    print(index)
    #    print(x.shape)
    #    print(y.shape)


if __name__ == "__main__":
    main()
