import os
import torch

import pandas as pd
import numpy as np

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True


def centered_iou(boxes_target, boxes_prediction):
    """
    This intersection over union function assumes that the center of the boxes are aligned. Should only be used to
    calculate the correct anchor box for the bbox
    :param boxes_target:
    :param boxes_prediction:
    :return:
    """
    intersection = torch.min(boxes_target[..., 0], boxes_prediction[..., 0]) * torch.min(boxes_target[..., 1], boxes_prediction[..., 1])
    union = (boxes_target[..., 0] * boxes_target[..., 1] + boxes_prediction[..., 0] * boxes_prediction[..., 1] - intersection)

    return intersection / union


def bboxes_to_target(bboxes, targets, ignore_iou_threshold, num_anchors_per_scale, anchors, Scale):
    """
    This function transforms the bboxes (labels) into a target structure. The target is a list of different sized 4D
    tensors e.g. list(torch.shape(3, 13, 13, 6), torch.shape(3, 26, 26, 6), torch.shape(3, 52, 52, 6))
    :param bboxes:
    :param targets:
    :param ignore_iou_threshold:
    :param num_anchors_per_scale:
    :param anchors:
    :param Scale:
    :return:
    """
    for box in bboxes:
        iou_anchors = centered_iou(box[2:4].clone().detach(), anchors)
        anchor_indices = iou_anchors.argsort(descending=True, dim=0)
        x, y, width, height, class_label = box
        has_anchor = [False, False, False]

        for anchor_idx in anchor_indices:
            scale_idx = anchor_idx // num_anchors_per_scale  # 0, 1, 2
            anchor_on_scale = anchor_idx % num_anchors_per_scale  # 0, 1, 2
            S = Scale[scale_idx]
            i, j = int(S * y), int(S * x)  # x = 0.5, S = 13 --> int(6.5) = 6
            anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

            if not anchor_taken and not has_anchor[scale_idx]:
                targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                x_cell, y_cell = S * x - j, S * y - i  # Both are between [0, 1]
                width_cell, height_cell = (width * S, height * S)
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                has_anchor[scale_idx] = True

            elif not anchor_taken and iou_anchors[anchor_idx] > ignore_iou_threshold:
                targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # Ignore this prediction

    return targets


class YoloDataset(Dataset):
    def __init__(self, csv_file, image_dir, label_dir, transforms, anchor_boxes, number_of_anchors, number_of_scales, ignore_iou_threshold, S=[13, 26, 52], C=20):
        self.annotations = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.anchor_boxes = anchor_boxes
        self.number_of_anchors = number_of_anchors
        self.number_of_scales = number_of_scales
        self.ignore_iou_threshold = ignore_iou_threshold
        self.S = S
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()  # [x, y, w, h, c]
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        argumentations = self.transforms(image=image, bboxes=bboxes)

        image = argumentations["image"]
        bboxes = argumentations["boxes"]

        targets = [torch.zeros((self.number_of_scales, scale, scale, 6)) for scale in self.S]

        targets = self.bboxes_to_target(bboxes, targets)

        return image, tuple(targets)





