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
        #A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.6),
        A.ShiftScaleRotate(rotate_limit=10, p=0.4, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        #A.Blur(p=0.2),
        #A.CLAHE(p=0.2),
        #A.Posterize(p=0.2),
        #A.ToGray(p=0.1),
        ToTensorV2()
    ],
    bbox_params=A.BboxParams(format="coco", min_visibility=0.4, label_fields=[])
)

COCO_CLASSES = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}

