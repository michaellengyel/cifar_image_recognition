import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import cv2

DATASET = '../datasets/voc/'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
BATCH_SIZE = 16
IMAGE_SIZE = 416
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 1000
CONF_THRESHOLD = 0.6
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45  # Non Max Suppression for bounding box removal
Scale = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]  # 13, 26, 52
C = 20
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL = True
SAVE_FREQUENCY = 10
CHECKPOINT_FILE = "checkpoint.pth.tar"
TRAINED_FILE = "checkpoint_trained.pth.tar"
IMG_DIR = DATASET + "images/"
LABEL_DIR = DATASET + "labels/"

# Rescaled anchors to be between [0, 1]
# These are calculated using k-means on COCO dataset
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)]
]

"""
train_transforms = A.Compose(
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
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[])
)

test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(min_height=int(IMAGE_SIZE), min_width=int(IMAGE_SIZE), border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        A.ShiftScaleRotate(rotate_limit=25, p=0.4, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2()
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[])
)
"""

min_transforms = A.Compose(
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
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[])
)

max_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE)),
        A.PadIfNeeded(min_height=int(IMAGE_SIZE), min_width=int(IMAGE_SIZE), border_mode=cv2.BORDER_CONSTANT),
        ToTensorV2()
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[])
)

PASCAL_CLASSES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
    'none'
]
