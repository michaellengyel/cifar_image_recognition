import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import cv2

coco = "/home/ad.adasworks.com/peter.lengyel/PycharmProjects/ml_practice/datasets/coco/"

root_train = coco + "images/train2017/"
annFile_train = coco + "annotations/instances_train2017.json"
root_val = coco + "images/val2017/"
annFile_val = coco + "annotations/instances_val2017.json"

IMAGE_SIZE = 416

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